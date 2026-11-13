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
        metadata = {**self.metadata, "kind": "newsclippings_full_test_baseline", "protocol": {"contract": {"prompt": self.metadata["prompt"]}},
                    "runner_sha256": sha256_file(Path(module.__file__)), "manifest_sha256": sha256_file(manifest_path),
                    "input_manifest_sha256": fingerprint(manifest), "sessions": [], "recoveries": []}
        write_json(self.directory / "run.json", metadata)
        first = FakeClient([response(error="offline")])
        second = FakeClient([response("MISMATCH"), response("MATCH"), response("MISMATCH")])
        with patch.object(module, "freeze_protocol", return_value={"contract": {"prompt": self.metadata["prompt"]}}), \
             patch.object(module, "checked_manifest", return_value=(manifest_path, manifest)), \
             patch.object(module, "check_server", side_effect=[(first, {}), (second, {})]) as probe:
            self.assertEqual(run_baseline(self.directory, config, root=self.root), 1)
            prefix = self.journal.read_bytes()
            with self.assertRaises(ValueError):
                run_baseline(self.directory, config, root=self.root)
            self.assertEqual(run_baseline(self.directory, config, resume=True, root=self.root), 0)
            self.assertTrue(self.journal.read_bytes().startswith(prefix))
            complete_bytes = self.journal.read_bytes()
            self.assertEqual(run_baseline(self.directory, config, resume=True, root=self.root), 0)
            self.assertEqual(self.journal.read_bytes(), complete_bytes)
            self.assertEqual(probe.call_count, 2)
        self.assertEqual(len(load_journal(self.journal, self.rows)[0]), 4)

    def test_new_run_initializes_without_any_old_results(self):
        import newsclip_pilot.baseline as module
        directory = self.root / "brand_new"
        manifest = {"records": self.rows}
        manifest_path = self.root / "prepared.json"
        write_json(manifest_path, manifest)
        protocol = {"contract": {"prompt": self.metadata["prompt"]}}
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

    def test_audited_legacy_runner_can_resume_without_changing_protocol(self):
        import newsclip_pilot.baseline as module
        from newsclip_pilot.data import sha256_file
        manifest = {"records": self.rows}
        manifest_path = self.root / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(self.directory / "input_manifest.json", manifest)
        protocol = {"contract": {"prompt": self.metadata["prompt"]}}
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
