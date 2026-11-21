"""Crash/resume tests use a fake client; no inference or external network calls."""

import json
import io
import os
import tarfile
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsclip_pilot.baseline import (StopRequest, append_result, exclusive_run, freeze_protocol,
                                    load_journal, process_pending, run_baseline, fingerprint)
from newsclip_pilot.client import interpret
from newsclip_pilot.data import prepare, select_test_records, write_json
from newsclip_pilot.evaluation import evaluate


def response(text="MATCH", error=None):
    return {"error": error, "response": None if error else {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
            "raw_http_response": text, "http_status": None if error else 200, "elapsed_seconds": 0.1}


def saved(row, reply=None):
    reply = response() if reply is None else reply
    return {**{key: row[key] for key in ("annotation_index", "pair_index", "id", "image_id", "true_label")},
            "started_at_utc": "fixture", **reply, **interpret(reply)}


class FakeClient:
    def __init__(self, replies, stop=None):
        self.replies = iter(replies)
        self.calls = []
        self.stop = stop

    def request(self, endpoint, payload):
        self.calls.append(payload)
        result = next(self.replies)
        if self.stop is not None:
            self.stop.requested = True
        return result


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "results"
        self.directory.mkdir()
        self.journal = self.directory / "predictions.jsonl"
        self.journal.touch()
        (self.root / "image.jpg").write_bytes(b"\xff\xd8\xff fixture bytes, never sent to a server")
        self.rows = [{"annotation_index": i, "pair_index": i // 2, "id": i // 2 + 100,
                      "image_id": i // 2 + 100 if i % 2 == 0 else 1000 + i, "true_label": "MATCH" if i % 2 == 0 else "MISMATCH",
                      "image_path": "image.jpg", "caption": f"Fixture caption {i}"} for i in range(4)]
        self.metadata = {"expected_records": 4, "prompt": "Fixed fixture prompt.", "config": {
            "model": "fixture", "temperature": 0, "seed": 42, "max_tokens": 16,
            "repeat_penalty": 1, "chat_template_kwargs": {"enable_thinking": False}}}

    def test_ctrl_c_style_stop_commits_current_then_resume_skips_it(self):
        stop = StopRequest()
        client = FakeClient([response()], stop)
        process_pending(self.directory, self.metadata, self.rows, [], client, stop, self.root)
        records, recovery = load_journal(self.journal, self.rows)
        self.assertEqual(len(records), 1)
        self.assertFalse(read_metrics(self.directory)["complete"])
        self.assertEqual(self.metadata["status"], "interrupted")
        self.assertIsNone(recovery)
        client = FakeClient([response("MISMATCH"), response("MATCH"), response("MISMATCH")])
        process_pending(self.directory, self.metadata, self.rows, records, client, StopRequest(), self.root)
        final, _ = load_journal(self.journal, self.rows)
        self.assertEqual([r["annotation_index"] for r in final], [0, 1, 2, 3])
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(read_metrics(self.directory)["complete"])
        client = FakeClient([])
        process_pending(self.directory, self.metadata, self.rows, final, client, StopRequest(), self.root)
        self.assertEqual(client.calls, [])

    def test_invalid_and_request_error_are_saved_skipped_and_counted(self):
        client = FakeClient([response(), response("MISMATCH."), response(error="timeout")])
        process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
        records, _ = load_journal(self.journal, self.rows)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.metadata["status"], "stopped_on_request_error")
        client = FakeClient([response("MISMATCH")])
        process_pending(self.directory, self.metadata, self.rows, records, client, StopRequest(), self.root)
        final, _ = load_journal(self.journal, self.rows)
        metrics = evaluate(final, 4)
        self.assertEqual(metrics["invalid_responses"], 1)
        self.assertEqual(metrics["request_errors"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(len(client.calls), 1)

    def test_sample_specific_400_continues_without_retry_or_losing_errors(self):
        errors = []
        for error in ({"type": "exceed_context_size_error", "message": "request (4204 tokens) exceeds context (4096)"},
                      {"type": "invalid_request_error", "message": "Failed to load image or audio file"}):
            item = response(error="HTTP 400: Bad Request")
            item.update(http_status=400, raw_http_response=json.dumps({"error": error}))
            errors.append(item)
        client = FakeClient([response(), errors[0], errors[1], response("MISMATCH")])
        process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
        records, _ = load_journal(self.journal, self.rows)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual([r["annotation_index"] for r in records], [0, 1, 2, 3])
        self.assertEqual([r.get("request_error_kind") for r in records], [None, "context_limit", "media_decode", None])
        metrics = read_metrics(self.directory)
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["request_errors"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        resumed = FakeClient([])
        process_pending(self.directory, self.metadata, self.rows, records, resumed, StopRequest(), self.root)
        self.assertEqual(resumed.calls, [])

    def test_unknown_400_still_stops(self):
        item = response(error="HTTP 400: Bad Request")
        item.update(http_status=400, raw_http_response=json.dumps({"error": {"message": "Invalid grammar"}}))
        client = FakeClient([item])
        process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(read_metrics(self.directory)["complete"])
        self.assertEqual(self.metadata["status"], "stopped_on_request_error")

    def test_connection_recovers_without_retrying_failed_example(self):
        import newsclip_pilot.baseline as module
        self.metadata["config"].update(base_url="http://127.0.0.1:8080", timeout_seconds=180)
        self.metadata["server"] = {"identity": "verified"}
        failure = response(error="URLError: <urlopen error [WinError 10060] connection timeout>")
        original = FakeClient([failure])
        recovered = FakeClient([response("MISMATCH"), response(), response("MISMATCH")])
        with patch.object(module.LocalClient, "request", return_value={"response": {"status": "ok"}, "error": None, "http_status": 200}) as health, \
             patch.object(module, "check_server", return_value=(recovered, self.metadata["server"])) as verify:
            process_pending(self.directory, self.metadata, self.rows, [], original, StopRequest(), self.root)
        self.assertEqual([call.args[0] for call in health.call_args_list], ["/health"])
        self.assertEqual(verify.call_args.args[2], {"identity": "verified"})
        self.assertEqual(recovered.timeout, 180)
        self.assertEqual(len(original.calls), 1)
        self.assertEqual(len(recovered.calls), 3)
        records, _ = load_journal(self.journal, self.rows)
        self.assertEqual([r["annotation_index"] for r in records], [0, 1, 2, 3])
        self.assertEqual(records[0]["request_error_kind"], "transport")
        self.assertTrue(read_metrics(self.directory)["complete"])
        self.assertEqual(read_metrics(self.directory)["request_errors"], 1)

    def test_connection_recovery_is_bounded_and_does_not_fill_dataset_with_errors(self):
        import newsclip_pilot.baseline as module
        self.metadata["config"].update(base_url="http://127.0.0.1:8080", timeout_seconds=180)
        self.metadata["server"] = {"identity": "verified"}
        failure = response(error="URLError: [WinError 10060] timeout")
        client = FakeClient([failure])
        with patch.object(module.LocalClient, "request", return_value=failure) as health, \
             patch.object(module, "check_server") as verify, patch.object(module.time, "sleep"):
            process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
        self.assertEqual(health.call_count, 4)
        verify.assert_not_called()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(read_metrics(self.directory)["n_evaluated"], 1)
        self.assertFalse(read_metrics(self.directory)["complete"])
        self.assertFalse(self.metadata["last_connection_recovery"]["recovered"])

    def test_connection_recovery_rejects_another_model(self):
        import newsclip_pilot.baseline as module
        self.metadata["config"].update(base_url="http://127.0.0.1:8080", timeout_seconds=180)
        self.metadata["server"] = {"identity": "verified"}
        client = FakeClient([response(error="URLError: connection reset")])
        with patch.object(module.LocalClient, "request", return_value={"response": {"status": "ok"}, "error": None}), \
             patch.object(module, "check_server", side_effect=RuntimeError("Different loaded model")):
            process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(read_metrics(self.directory)["n_evaluated"], 1)
        self.assertIn("Different loaded model", self.metadata["last_connection_recovery"]["error"])

    def test_partial_final_write_is_backed_up_and_previous_records_preserved(self):
        with self.journal.open("ab") as handle:
            append_result(handle, saved(self.rows[0]))
            handle.write(b'{"annotation_index": 1, "raw_response": "partial')
        records, recovery = load_journal(self.journal, self.rows)
        self.assertEqual(len(records), 1)
        self.assertIn("incomplete", recovery["action"])
        self.assertTrue((self.directory / recovery["backup"]).read_bytes().endswith(b'"partial'))
        again, recovery = load_journal(self.journal, self.rows)
        self.assertEqual(records, again)
        self.assertIsNone(recovery)

    def test_valid_final_record_without_newline_is_not_lost(self):
        self.journal.write_bytes(json.dumps(saved(self.rows[0])).encode())
        records, recovery = load_journal(self.journal, self.rows)
        self.assertEqual(len(records), 1)
        self.assertEqual(recovery["action"], "added_missing_final_newline")
        self.assertTrue(self.journal.read_bytes().endswith(b"\n"))

    def test_duplicate_or_corrupt_complete_line_is_rejected_without_rewriting(self):
        for data in ((json.dumps(saved(self.rows[0])) + "\n") * 2,
                     json.dumps(saved(self.rows[0])) + '\n{"broken":\n'):
            self.journal.write_bytes(data.encode())
            before = self.journal.read_bytes()
            with self.assertRaises(ValueError):
                load_journal(self.journal, self.rows)
            self.assertEqual(before, self.journal.read_bytes())

    def test_changed_label_or_saved_parse_is_rejected(self):
        for field, value in (("true_label", "MISMATCH"), ("prediction", "MISMATCH")):
            row = saved(self.rows[0])
            row[field] = value
            self.journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_journal(self.journal, self.rows)

    def test_os_lock_rejects_second_writer_and_releases_after_exit(self):
        with exclusive_run(self.directory):
            with self.assertRaises(RuntimeError):
                with exclusive_run(self.directory):
                    self.fail("Second writer acquired lock")
        with exclusive_run(self.directory):
            pass

    def test_run_resume_entrypoint_and_completed_noop(self):
        import newsclip_pilot.baseline as module
        from newsclip_pilot.data import sha256_file
        manifest = {"records": self.rows}
        manifest_path = self.root / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(self.directory / "input_manifest.json", manifest)
        config = self.metadata["config"]
        protocol = {"contract": {"prompt": self.metadata["prompt"]}, "fingerprint": fingerprint({"prompt": self.metadata["prompt"]})}
        metadata = {**self.metadata, "kind": "newsclippings_full_test_baseline", "protocol": protocol,
                    "runner_sha256": sha256_file(Path(module.__file__)), "manifest_sha256": sha256_file(manifest_path),
                    "input_manifest_sha256": fingerprint(manifest), "sessions": [], "recoveries": []}
        write_json(self.directory / "run.json", metadata)
        first = FakeClient([response(error="offline")])
        second = FakeClient([response("MISMATCH"), response("MATCH"), response("MISMATCH")])
        with patch.object(module, "freeze_protocol", return_value=protocol), \
             patch.object(module, "checked_manifest", return_value=(manifest_path, manifest)), \
             patch.object(module, "check_server", side_effect=[(first, {}), (second, {})]) as probe, \
             patch("newsclip_pilot.reports.refresh_after_run") as reports:
            self.assertEqual(run_baseline(self.directory, config, root=self.root), 1)
            prefix = self.journal.read_bytes()
            with self.assertRaises(ValueError):
                run_baseline(self.directory, config, root=self.root)
            with self.assertRaisesRegex(ValueError, "saved snapshot"):
                run_baseline(self.directory, {**config, "model": "another-model"}, resume=True, root=self.root)
            for extra in ({"experiment":{"context_size":8192}},
                          {"experiment":{"image_processing":{"image_max_tokens":8359}}}):
                with self.assertRaisesRegex(ValueError,"saved snapshot"):
                    run_baseline(self.directory,{**config,**extra},resume=True,root=self.root)
            self.assertEqual(run_baseline(self.directory, config, resume=True, root=self.root), 0)
            self.assertTrue(self.journal.read_bytes().startswith(prefix))
            complete_bytes = self.journal.read_bytes()
            self.assertEqual(run_baseline(self.directory, config, resume=True, root=self.root), 0)
            self.assertEqual(self.journal.read_bytes(), complete_bytes)
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(reports.call_count, 3)  # stopped run, resumed completion, completed resume
        self.assertEqual(len(load_journal(self.journal, self.rows)[0]), 4)

    def test_run_automatically_checks_separate_val_before_test(self):
        import newsclip_pilot.baseline as module
        config={**self.metadata["config"],"split":"test","validation":{"run_dir":"val"}}
        manifest={"records":self.rows}
        path=self.root/"prepared.json";write_json(path,manifest)
        protocol={"contract":{"prompt":self.metadata["prompt"]},"fingerprint":fingerprint({"prompt":self.metadata["prompt"]})}
        test_client=FakeClient([response() for _ in self.rows])
        val_client=FakeClient([response() for _ in self.rows])
        with patch.object(module,"freeze_protocol",return_value=protocol), \
             patch.object(module,"checked_manifest",return_value=(path,manifest)), \
             patch.object(module,"check_server",side_effect=[(test_client,{}),(val_client,{})]) as probe, \
             patch("newsclip_pilot.experiments.validation_config",return_value={**config,"split":"val"}), \
             patch("newsclip_pilot.experiments.require_validation",return_value={"n_evaluated":4}) as gate:
            self.assertEqual(run_baseline(self.root/"test",config,root=self.root),0)
            self.assertEqual(gate.call_count,1)
            self.assertEqual(len(val_client.calls),4)
            self.assertEqual(len(test_client.calls),4)
            self.assertEqual(run_baseline(self.root/"test",config,resume=True,root=self.root),0)
            self.assertEqual(probe.call_count,2)
        for directory in (self.root/"val",self.root/"test"):
            self.assertEqual(len(load_journal(directory/"predictions.jsonl",self.rows)[0]),4)
        stop=StopRequest()
        interrupted={**config,"validation":{"run_dir":"val_interrupted"}}
        client=FakeClient([response()],stop=stop)
        with patch.object(module,"freeze_protocol",return_value=protocol), \
             patch.object(module,"checked_manifest",return_value=(path,manifest)), \
             patch.object(module,"check_server",side_effect=[(test_client,{}),(client,{})]), \
             patch("newsclip_pilot.experiments.validation_config",return_value={**interrupted,"split":"val"}):
            self.assertEqual(run_baseline(self.root/"interrupted",interrupted,root=self.root,stop_request=stop),130)
        self.assertEqual((self.root/"interrupted/predictions.jsonl").read_bytes(),b"")

    def test_new_run_initializes_without_any_old_results(self):
        import newsclip_pilot.baseline as module
        directory = self.root / "brand_new"
        manifest = {"records": self.rows}
        manifest_path = self.root / "prepared.json"
        write_json(manifest_path, manifest)
        protocol = {"contract": {"prompt": self.metadata["prompt"]}, "fingerprint": fingerprint({"prompt": self.metadata["prompt"]})}
        client = FakeClient([response(error="offline")])
        with patch.object(module, "freeze_protocol", return_value=protocol), \
             patch.object(module, "checked_manifest", return_value=(manifest_path, manifest)), \
             patch.object(module, "check_server", return_value=(client, {})):
            self.assertEqual(run_baseline(directory, self.metadata["config"], root=self.root), 1)
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["n_completed"], 1)
        self.assertEqual(metadata["protocol"], protocol)
        self.assertNotIn("reference_pilot", metadata["config"])

    @unittest.skipUnless(os.name == "nt", "Windows rename/sharing regression")
    def test_real_windows_metrics_lock_does_not_stop_or_repeat_requests(self):
        metrics_path = self.directory / "metrics.json"
        write_json(metrics_path, evaluate([], 4))
        client = FakeClient([response(), response("MISMATCH"), response(), response("MISMATCH")])
        # A real Windows reader without FILE_SHARE_DELETE reproduces the saved WinError 5.
        with metrics_path.open("rb"), patch("newsclip_pilot.storage.time.sleep"):
            result = process_pending(self.directory, self.metadata, self.rows, [], client, StopRequest(), self.root)
            self.assertEqual(result["n_evaluated"], 4)
            self.assertEqual(read_metrics(self.directory)["n_evaluated"], 0)
        records, _ = load_journal(self.journal, self.rows)
        resumed = FakeClient([])
        process_pending(self.directory, self.metadata, self.rows, records, resumed, StopRequest(), self.root)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(read_metrics(self.directory)["n_evaluated"], 4)
        events = [json.loads(line) for line in (self.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIn("snapshot_deferred", {e["event"] for e in events})
        self.assertIn("snapshot_recovered", {e["event"] for e in events})

    def test_shortened_queue_cannot_be_reported_as_complete(self):
        client = FakeClient([])
        with self.assertRaisesRegex(ValueError, "every expected unique"):
            process_pending(self.directory, self.metadata, self.rows[:3], [], client, StopRequest(), self.root)
        self.assertEqual(client.calls, [])
        self.assertFalse(evaluate([saved(r) for r in self.rows[:3]], 4)["complete"])

    def test_read_only_status_never_repairs_journal(self):
        original = json.dumps(saved(self.rows[0])).encode()
        self.journal.write_bytes(original)
        records, _ = load_journal(self.journal, self.rows, repair=False)
        self.assertEqual(len(records), 1)
        self.assertEqual(self.journal.read_bytes(), original)
        self.journal.write_bytes(original + b'\n{"incomplete":')
        with self.assertRaises(ValueError):
            load_journal(self.journal, self.rows, repair=False)
        self.assertTrue(self.journal.read_bytes().endswith(b'{"incomplete":'))

    def test_val_gate_requires_same_model_but_never_selects_by_accuracy(self):
        from newsclip_pilot.experiments import require_validation
        protocol = {"contract": {"fixture": "fixed"}, "fingerprint": fingerprint({"fixture": "fixed"})}
        identity = {"model": "A"}
        write_json(self.directory / "run.json", {"split": "val", "protocol": protocol, "server": {"identity": identity}})
        write_json(self.directory / "input_manifest.json", {"records": self.rows})
        # Every answer is wrong, one is INVALID: neither is silently filtered.
        answers = ["MISMATCH", "MATCH", "MATCH.", "MATCH"]
        with self.journal.open("ab") as handle:
            for row, answer in zip(self.rows, answers):
                append_result(handle, saved(row, response(answer)))
        config = {"validation": {"run_dir": "results"}}
        with patch("newsclip_pilot.baseline.freeze_protocol", return_value=protocol), \
             patch("newsclip_pilot.experiments.validation_config", return_value=config), \
             patch("newsclip_pilot.experiments.validation_manifest", return_value=(None, {"records": self.rows})):
            info = require_validation(config, self.root, {"identity": identity})
            self.assertEqual(info["n_evaluated"], 4)
            self.assertEqual(info["invalid_responses"], 1)
            with self.assertRaisesRegex(RuntimeError, "another model"):
                require_validation(config, self.root, {"identity": {"model": "B"}})
            with self.journal.open("wb") as handle:
                for row in self.rows:
                    append_result(handle, saved(row, response("<think></think>MATCH")))
            with self.assertRaisesRegex(RuntimeError, "no exact MATCH/MISMATCH"):
                require_validation(config, self.root, {"identity": identity})

    def test_audited_legacy_runner_can_resume_without_changing_protocol(self):
        import newsclip_pilot.baseline as module
        from newsclip_pilot.data import sha256_file
        manifest = {"records": self.rows}
        manifest_path = self.root / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(self.directory / "input_manifest.json", manifest)
        protocol = {"contract": {"prompt": self.metadata["prompt"]}, "fingerprint": fingerprint({"prompt": self.metadata["prompt"]})}
        metadata = {**self.metadata, "kind": "newsclippings_full_test_baseline", "protocol": protocol,
                    "runner_sha256": module.LEGACY_RUNNER_SHA256, "manifest_sha256": sha256_file(manifest_path),
                    "input_manifest_sha256": fingerprint(manifest), "sessions": [], "recoveries": []}
        write_json(self.directory / "run.json", metadata)
        with self.journal.open("ab") as handle:
            append_result(handle, saved(self.rows[0]))
        prefix = self.journal.read_bytes()
        client = FakeClient([response("MISMATCH"), response(), response("MISMATCH")])
        with patch.object(module, "freeze_protocol", return_value=protocol), \
             patch.object(module, "checked_manifest", return_value=(manifest_path, manifest)), \
             patch.object(module, "check_server", return_value=(client, {})):
            self.assertEqual(run_baseline(self.directory, self.metadata["config"], resume=True, root=self.root), 0)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(self.journal.read_bytes().startswith(prefix))
        updated = json.loads((self.directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(updated["protocol"], protocol)
        self.assertEqual(updated["runner_sha256"], module.LEGACY_RUNNER_SHA256)
        self.assertEqual(updated["active_runner_sha256"], sha256_file(Path(module.__file__)))


class MappingAndCountTests(unittest.TestCase):
    def test_caption_by_id_and_image_by_image_id_for_both_pair_relations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(id=10, image_id=10, falsified=False, source_dataset=0),
                    dict(id=11, image_id=10, falsified=True, source_dataset=0),
                    dict(id=20, image_id=20, falsified=False, source_dataset=0),
                    dict(id=20, image_id=21, falsified=True, source_dataset=0)]
            write_json(root / "test.json", {"annotations": rows, "source_datasets": {"0": "fixture"}})
            metadata = [{"id": i, "caption": f"Caption {i}", "image_path": f"./images/{i}.jpg"} for i in (10, 11, 20, 21)]
            with tarfile.open(root / "origin.tar", "w") as archive:
                files = {f"origin/images/{i}.jpg": f"Image {i}".encode() for i in (10, 20, 21, 999)}
                files["origin/data.json"] = json.dumps(metadata).encode()
                for name, content in files.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            config = {"annotations": "test.json", "archive": "origin.tar", "prepared_dir": "prepared", "seed": 42, "expected_records": 4}
            manifest = prepare(root, config)
            self.assertEqual(len(manifest["records"]), 4)
            self.assertEqual(manifest["unique_images"], 3)
            for row in manifest["records"]:
                self.assertEqual(row["caption"], f"Caption {row['id']}")
                self.assertEqual((root / row["image_path"]).read_bytes(), f"Image {row['image_id']}".encode())
            self.assertFalse((root / "origin").exists())
            self.assertEqual(len(list((root / "prepared/images").iterdir())), 3)

    def test_all_attempts_in_denominator_and_incomplete_count(self):
        cases = [("MATCH", "MATCH", "ok"), ("MATCH", "MISMATCH", "ok"), ("MATCH", None, "invalid_response"),
                 ("MISMATCH", "MISMATCH", "ok"), ("MISMATCH", "MATCH", "ok"), ("MISMATCH", None, "request_error")]
        rows = [dict(annotation_index=i, true_label=truth, prediction=pred, status=status)
                for i, (truth, pred, status) in enumerate(cases)]
        result = evaluate(rows, 8)
        self.assertEqual(result["n_evaluated"], 6)
        self.assertFalse(result["complete"])
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], 0.4)
        self.assertEqual(result["confusion_matrix_with_failures"]["values"], [[1, 1, 1, 0], [1, 1, 0, 1]])


def read_metrics(directory):
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


class FrozenProtocolTests(unittest.TestCase):
    def test_capacity_cache_bounds_identity_and_undecodable_records(self):
        from newsclip_pilot.capacity import digest_file,validate_capacity_index
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ("llama.dll","mtmd.dll"):(root/name).write_bytes(b"fixture DLL; never loaded")
            manifest={"records":[dict(annotation_index=i,image_sha256=f"image{i}",caption=f"Caption {i}") for i in range(2)]}
            prepared=root/"prepared";prepared.mkdir();write_json(prepared/"manifest.json",manifest)
            image={"client_resize":None,"image_min_tokens":-1,"image_max_tokens":-1}
            config={"model_id":"lfm3_vl_2b","prepared_dir":"prepared","max_tokens":16,"protocol":{"prompt_sha256":"prompt"},
                    "experiment":{"context_size":4352,"gguf":{"sha256":"weights"},"mmproj":[{"sha256":"projector"}],"image_processing":image},
                    "capacity_preflight":{"path":"counts.json","margin_tokens":64}}
            records=[dict(dataset="NewsCLIPpings",input_id=f"news:{i}",image_sha256=f"image{i}",decodable=i==0,
                          caption_sha256=hashlib.sha256(f"Caption {i}".encode()).hexdigest()) for i in range(2)]
            records[0].update(input_tokens=4221,input_positions=4221)
            proof={"gguf_sha256":"weights","mmproj_sha256":"projector","prompt_sha256":"prompt",**image,
                   "news_manifest_sha256":digest_file(prepared/"manifest.json"),
                   "dll_sha256":{name:digest_file(root/name) for name in ("llama.dll","mtmd.dll")}}
            index={"model":"lfm3_vl_2b","historical_mismatches":[],"proof":proof,"records":records}
            write_json(root/"counts.json",index);config["capacity_preflight"]["sha256"]=digest_file(root/"counts.json")
            before=json.dumps(manifest)
            with patch("newsclip_pilot.capacity.shutil.which",return_value=str(root/"llama.exe")):
                summary=validate_capacity_index(config,manifest,root)
                self.assertEqual(summary["counted_records"],1)
                self.assertEqual(summary["native_decoder_failures_without_length"],1)
                self.assertEqual(json.dumps(manifest),before)
                config["experiment"]["context_size"]=4096
                with self.assertRaisesRegex(ValueError,"annotation_index=0"):
                    validate_capacity_index(config,manifest,root)
                config["experiment"]["context_size"]=4352
                manifest["records"][0]["caption"]="changed"
                with self.assertRaisesRegex(ValueError,"input changed"):
                    validate_capacity_index(config,manifest,root)
                manifest["records"][0]["caption"]="Caption 0"
                (root/"counts.json").write_text("modified",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"cache changed"):
                    validate_capacity_index(config,manifest,root)

    def test_native_template_only_allows_one_terminal_lf_removal(self):
        import hashlib
        from newsclip_pilot.experiments import verify_native_template
        embedded = "native\n  template\n"
        pinned = hashlib.sha256(embedded.encode()).hexdigest()
        proof = verify_native_template(embedded, embedded[:-1], pinned)
        self.assertEqual(proof["normalization"], "one terminal LF removed by server")
        self.assertEqual(proof["embedded_sha256"], pinned)
        self.assertEqual(verify_native_template(embedded, embedded, pinned)["normalization"], "exact")
        for served in ("native\ntemplate", " native\n  template", embedded + "\n", "native\n  template "):
            with self.assertRaises(RuntimeError):
                verify_native_template(embedded, served, pinned)
        with self.assertRaises(RuntimeError):
            verify_native_template(embedded, embedded[:-1], "wrong hash")
        with self.assertRaises(RuntimeError):
            verify_native_template(embedded + "\n", embedded[:-1])

    def test_reports_distinguish_pending_incomplete_and_orphaned_answers(self):
        import newsclip_pilot.baseline as module
        from newsclip_pilot.reports import build_report, refresh_after_run
        import csv
        config = module.load_model_config(module.ROOT, model_id="lfm3_vl_2b")
        configs={model:module.load_model_config(module.ROOT,model_id=model) for model in module.MODEL_ALIASES}
        protocol = module.freeze_protocol(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / config["run_dir"]
            directory.mkdir(parents=True)
            path = root / config["prepared_dir"] / "manifest.json"
            rows = [dict(annotation_index=i, pair_index=i//2, id=i//2, image_id=i,
                         caption="fixture", image_sha256="fixture", true_label="MATCH" if i%2==0 else "MISMATCH")
                    for i in range(7264)]
            manifest = {"records": rows, "audit": {"records": 7264}}
            path.parent.mkdir(parents=True)
            write_json(path, manifest)
            write_json(path.parent / "selection.json", manifest)
            with patch.object(module, "checked_manifest", return_value=(path, manifest)), \
                 patch.object(module, "freeze_protocol", return_value=protocol), \
                 patch("newsclip_pilot.reports.freeze_protocol", return_value=protocol), \
                 patch("newsclip_pilot.client.LocalClient.request", side_effect=AssertionError("No HTTP")) as http:
                pending = build_report(directory, config, root)
                self.assertEqual(pending["status"], "pending")
                self.assertEqual(pending["n_evaluated"], 0)
                self.assertIsNone(pending["accuracy"])
                self.assertIsNone(pending["per_class"]["MATCH"]["f1"])
                self.assertIsNone(pending["source_run_id"])
                self.assertIsNone(pending["protocol"])
                self.assertFalse(pending["complete"])
                module.initialize(directory, config, protocol, path, manifest, root)
                with (directory / "predictions.jsonl").open("ab") as handle:
                    append_result(handle, saved(rows[0], response(error="fixture infrastructure error")))
                incomplete = build_report(directory, config, root)
                self.assertEqual(incomplete["status"], "incomplete")
                self.assertEqual(incomplete["n_evaluated"], 1)
                self.assertEqual(incomplete["request_errors"], 1)
                self.assertEqual(incomplete["accuracy"], 0.0)
                self.assertFalse(incomplete["complete"])
                self.assertFalse(incomplete["historical_metrics_match"])
                prefix=(directory/"predictions.jsonl").read_bytes()
                with patch("newsclip_pilot.reports.load_model_config",side_effect=lambda root,model_id:configs[model_id]):
                    with patch("newsclip_pilot.reports.write_text",side_effect=PermissionError("CSV locked")), \
                         patch("sys.stderr",new=io.StringIO()) as warning:
                        refresh_after_run(directory,config,root)
                    self.assertIn("compare",warning.getvalue())
                    self.assertIn("Saved responses",warning.getvalue())
                    self.assertEqual((directory/"predictions.jsonl").read_bytes(),prefix)
                    refresh_after_run(directory,config,root)
                    self.assertNotIn(b"\r\r\n",(root/"reports/model_comparison.csv").read_bytes())
                    comparison=list(csv.DictReader((root/"reports/model_comparison.csv").open(encoding="utf-8")))
                    self.assertEqual([r["status"] for r in comparison],["incomplete","pending","pending"])
                    self.assertEqual(comparison[1]["accuracy"],"")
                    with (directory/"predictions.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write("".join(json.dumps(saved(row))+"\n" for row in rows[1:]))
                    complete_bytes=(directory/"predictions.jsonl").read_bytes()
                    refresh_after_run(directory,config,root)
                    result=json.loads((root/"reports/lfm3_vl_2b/metrics.json").read_text())
                    self.assertEqual(result["status"],"complete")
                    self.assertEqual(result["n_evaluated"],7264)
                    self.assertTrue(result["complete"])
                    self.assertEqual((directory/"predictions.jsonl").read_bytes(),complete_bytes)
                orphan = root / "results/orphan"
                orphan.mkdir()
                (orphan / "predictions.jsonl").write_text("old answers", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "no run.json"):
                    module.inspect_run(orphan, config, root)
                self.assertEqual(http.call_count, 0)

    def test_cli_uses_separate_model_defaults_and_keeps_explicit_directory(self):
        import newsclip_pilot.baseline as module
        for identifier in module.MODEL_ALIASES:
            file = f"configs/test_{identifier}.json"
            with patch.object(module, "inspect_run", return_value={}) as inspect:
                self.assertEqual(module.main(["--config", file, "status"]), 0)
                self.assertEqual(inspect.call_args.args[0].name, f"test_merged_balanced_{identifier}")
                self.assertEqual(module.main(["--model", identifier, "status"]), 0)
                self.assertEqual(inspect.call_args.args[0].name, f"test_merged_balanced_{identifier}")
                self.assertEqual(module.main(["--config", file, "status", "--run-dir", "results/custom"]), 0)
                self.assertEqual(inspect.call_args.args[0].name, "custom")

    def test_explicit_model_required_and_alias_must_match(self):
        import newsclip_pilot.baseline as module
        with self.assertRaises(SystemExit), patch("sys.stderr", new=io.StringIO()):
            module.main(["status"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "bad.json", {"model_id": "lfm3_vl_2b", "model": module.MODEL_ALIASES["qwen35_2b"]})
            with self.assertRaisesRegex(ValueError, "disagree"):
                module.load_model_config(root, path="bad.json")

    def test_audited_protocol_upgrade_preserves_snapshots_and_rejects_drift(self):
        from newsclip_pilot.baseline import PREVIOUS_HELPER_SHA256, protocols_match
        old = {"orchestration_sha256": PREVIOUS_HELPER_SHA256, "prompt": "fixed", "generation_config": {"max_tokens": 16}}
        new = {**old, "orchestration_sha256": "current verified helper"}
        snapshot = lambda c: {"contract": c, "fingerprint": fingerprint(c)}
        saved_protocol = snapshot(old)
        before = json.dumps(saved_protocol)
        self.assertTrue(protocols_match(saved_protocol, snapshot(new)))
        self.assertEqual(before, json.dumps(saved_protocol))
        for changed in ({**new, "prompt": "changed"}, {**new, "generation_config": {"max_tokens": 32}}):
            self.assertFalse(protocols_match(saved_protocol, snapshot(changed)))
        self.assertFalse(protocols_match({**saved_protocol, "fingerprint": "corrupt"}, snapshot(new)))

    def test_weight_hash_native_template_context_and_projector_are_checked(self):
        import struct
        from newsclip_pilot.data import sha256_file
        from newsclip_pilot.experiments import verify_model_server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def gguf(name):
                path = root / name
                key, value = b"tokenizer.chat_template", b"native template"
                path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, len(key)) + key +
                                 struct.pack("<IQ", 8, len(value)) + value)
                return path
            model, projector = gguf("model.gguf"), gguf("mmproj.gguf")
            (root / "server.log").write_text(f"srv load_model: loaded multimodal model, '{projector}'\n")
            experiment = dict(original_model_id="fixture/model", gguf_repository="fixture/gguf", revision="revision",
                              gguf=dict(filename=model.name, sha256=sha256_file(model)),
                              mmproj=[dict(filename=projector.name, sha256=sha256_file(projector))],
                              context_size=4096, reasoning_budget=0, launch_command=["fixture"])
            config = dict(experiment=experiment, base_url="http://127.0.0.1:8080", server_log="server.log")
            props = dict(model_path=str(model), modalities={"vision": True}, chat_template="native template",
                         total_slots=1, default_generation_settings={"n_ctx": 4096}, build_info="build")
            process = {"command_line": f'llama serve --log-file "{root / "server.log"}" --reasoning-budget 0', "pid": 123}
            def server():
                return {"props": {"response": json.loads(json.dumps(props))}}
            with patch("newsclip_pilot.experiments.observe_process", return_value=process):
                verified = verify_model_server(config, server(), root)
                self.assertEqual(verified["identity"]["mmproj_sha256"], sha256_file(projector))
                experiment["response_reasoning_format"] = "deepseek"
                with self.assertRaisesRegex(RuntimeError, "--reasoning-format deepseek"):
                    verify_model_server(config, server(), root)
                process["command_line"] += " --reasoning-format deepseek"
                verified = verify_model_server(config, server(), root)
                self.assertEqual(verified["identity"]["response_reasoning_format"], "deepseek")
                experiment["server_requirements"] = {"build_info": "build", "default_generation_settings": props["default_generation_settings"]}
                changed = server(); changed["props"]["response"]["build_info"] = "another build"
                with self.assertRaisesRegex(RuntimeError, "build_info"):
                    verify_model_server(config, changed, root)
                changed = server(); changed["props"]["response"]["default_generation_settings"]["params"] = {"top_k": 50}
                with self.assertRaisesRegex(RuntimeError, "default_generation_settings"):
                    verify_model_server(config, changed, root)
                changed = server(); changed["props"]["response"]["chat_template"] = "LFM template"
                with self.assertRaisesRegex(RuntimeError, "native template"):
                    verify_model_server(config, changed, root)
                changed = server(); changed["props"]["response"]["total_slots"] = 2
                with self.assertRaisesRegex(RuntimeError, "parallel 1"):
                    verify_model_server(config, changed, root)
                model.write_bytes(model.read_bytes() + b"different weights")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    verify_model_server(config, server(), root)

    def test_server_validation_without_results_and_relocated_weights(self):
        import newsclip_pilot.baseline as module
        props = dict(model_alias="fixture", model_ftype="Q4", model_path="another/place/model.gguf",
                     build_info="build", chat_template="template", default_generation_settings={"params": {"temperature": 0}})
        config = {"base_url": "http://127.0.0.1:8080", "timeout_seconds": 10, "model": "fixture",
                  "protocol": {"server": module.server_signature(props)}}
        with patch.object(module.LocalClient, "probe", return_value={"props": {"response": props}}):
            module.check_server(config)
            saved = {"props": {"response": {**props, "model_path": "old/location/model.gguf"}}}
            module.check_server(config, saved_server=saved)
            with self.assertRaises(RuntimeError):
                module.check_server({**config, "protocol": {"server": {**config["protocol"]["server"], "model_ftype": "Q8"}}})

    def test_changed_prompt_or_parser_is_rejected_without_results(self):
        import hashlib
        from newsclip_pilot.baseline import GENERATION_KEYS, FROZEN_FILES
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompt.txt").write_text("Fixed prompt", encoding="utf-8")
            config = {key: 42 for key in GENERATION_KEYS}
            config.update(prompt_file="prompt.txt", protocol={
                "prompt_sha256": hashlib.sha256(b"Fixed prompt").hexdigest(),
                "code_sha256": {name: "unchanged" for name in FROZEN_FILES}})
            with patch("newsclip_pilot.baseline.sha256_file", return_value="unchanged"):
                freeze_protocol(config, root)
                self.assertFalse((root / "results").exists())
                (root / "prompt.txt").write_text("Changed prompt", encoding="utf-8")
                with self.assertRaises(ValueError):
                    freeze_protocol(config, root)
                (root / "prompt.txt").write_text("Fixed prompt", encoding="utf-8")
            with patch("newsclip_pilot.baseline.sha256_file", return_value="changed"):
                with self.assertRaises(ValueError):
                    freeze_protocol(config, root)

    def test_full_test_count_and_original_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.json"
            rows = []
            for i in range(3632):
                rows.extend([{"id": i, "image_id": i, "falsified": False, "source_dataset": 0},
                             {"id": i, "image_id": 10000 + i, "falsified": True, "source_dataset": 0}])
            write_json(path, {"annotations": rows, "source_datasets": {"0": "fixture"}})
            selected = select_test_records(path)
            self.assertEqual([r["annotation_index"] for r in selected["records"]], list(range(7264)))
            self.assertEqual(selected["class_counts"], {"MATCH": 3632, "MISMATCH": 3632})
            write_json(path, {"annotations": rows[:-2], "source_datasets": {"0": "fixture"}})
            with self.assertRaises(ValueError):
                select_test_records(path)


if __name__ == "__main__":
    unittest.main()
