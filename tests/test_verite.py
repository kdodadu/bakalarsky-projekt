"""Adapted binary VERITE checks; synthetic data and fake clients only."""
import base64
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multimodal_baseline import baseline as b
from multimodal_baseline.data import prepare, verite_records
from multimodal_baseline.reports import build_report, compare_results
from test_baseline import FakeClient, response


class VeriteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        project = Path(__file__).resolve().parents[1]
        for name in (*b.FROZEN_FILES, "configs/prompt.txt"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
        self.csv = self.root / "raw/verite/VERITE.csv"
        self.csv.parent.mkdir(parents=True)
        self.images = self.csv.parent / "images"
        (self.images / "images").mkdir(parents=True)
        self.image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
        for name in ("shared.png", "other.png"):
            (self.images / "images" / name).write_bytes(self.image_bytes)
        self.rows = [{"": identifier, "caption": f'Caption {index}, "quoted"\nsecond line',
                      "image_path": "images/shared.png" if index % 2 == 0 else "images/other.png",
                      "label": category, "verdict": "SECRET_FACTCHECK", "corrected_caption": "ALTERNATIVE_CAPTION"}
                     for index, (identifier, category) in enumerate(zip("abcdef", ("true", "miscaptioned", "out-of-context") * 2))]
        self.write_csv()
        self.configs = {}
        for model in b.MODEL_ALIASES:
            config = b.load_model_config(project, model_id=model, dataset="verite")
            config.update(annotations="raw/verite/VERITE.csv", images_dir="raw/verite/images", expected_records=6,
                          expected_source_counts={"true": 2, "miscaptioned": 2, "out-of-context": 2},
                          expected_binary_counts={"MATCH": 2, "MISMATCH": 4})
            self.configs[model] = config
            config_path = self.root / f"configs/verite/{model}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")
        self.config = self.configs["lfm3_vl_2b"]

    def write_csv(self):
        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["", "caption", "image_path", "label", "verdict", "corrected_caption"])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_csv_mapping_multiline_and_no_image_copies(self):
        manifest = prepare(self.root, self.config)
        self.assertEqual(manifest["class_counts"], {"MATCH": 2, "MISMATCH": 4})
        self.assertEqual([r["id"] for r in manifest["records"]], list("abcdef"))
        self.assertEqual(manifest["records"][0]["caption"], self.rows[0]["caption"])
        self.assertEqual([r["source_category"] for r in manifest["records"]], [r["label"] for r in self.rows])
        self.assertEqual(manifest["unique_images"], 2)
        self.assertEqual(manifest["audit"]["duplicate_image_caption_pairs"], 0)
        self.assertFalse((self.root / "data/processed/verite/images").exists())
        self.assertEqual((self.images / "images/shared.png").read_bytes(), self.image_bytes)
        self.assertNotIn("verdict", manifest["records"][0])

    def test_bad_labels_fields_duplicate_ids_and_missing_images_are_errors(self):
        original = copy.deepcopy(self.rows)
        for key, value, message in (("label", "unknown", "unknown source label"),
                                    ("", "b", "Duplicate VERITE source ID"),
                                    ("caption", "", "required fields"),
                                    ("image_path", "images/missing.png", "missing image")):
            with self.subTest(key=key):
                self.rows = copy.deepcopy(original)
                self.rows[0][key] = value
                self.write_csv()
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    verite_records(self.root, self.config)
        self.rows = original
        self.write_csv()
        wrong_count = {**self.config, "expected_records": 1001}
        with self.assertRaisesRegex(ValueError, "6 records, expected 1001"):
            verite_records(self.root, wrong_count)

    def test_resume_category_metrics_reports_and_offline_reconstruction(self):
        prepare(self.root, self.config)
        directory = self.root / self.config["run_dir"]
        stop = b.StopRequest()
        first = FakeClient([response()], stop)
        decode_error = response(error="HTTP 400: Bad Request")
        decode_error.update(http_status=400, raw_http_response=json.dumps({"error": {"type": "invalid_request_error", "message": "Failed to load image or audio file"}}))
        second = FakeClient([response("MISMATCH."), decode_error, response("MISMATCH"), response("MISMATCH"), response("MISMATCH")])
        with patch.object(b, "check_server", side_effect=[(first, {}), (second, {})]) as server:
            self.assertEqual(b.run_baseline(directory, self.config, root=self.root, stop_request=stop), 130)
            prefix = (directory / "predictions.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "mapping"):
                b.run_baseline(directory, {**self.config, "binary_label_mapping": {"true": "MISMATCH"}}, True, self.root)
            self.assertEqual(b.run_baseline(directory, self.config, True, self.root), 0)
            self.assertEqual(server.call_count, 2)
        self.assertTrue((directory / "predictions.jsonl").read_bytes().startswith(prefix))
        self.assertEqual(len(second.calls), 5)
        for payload, row in zip(first.calls + second.calls, self.rows):
            message = payload["messages"]
            self.assertEqual(len(message), 1)
            self.assertEqual([p["type"] for p in message[0]["content"]], ["image_url", "text"])
            self.assertEqual(message[0]["content"][1]["text"], (self.root / "configs/prompt.txt").read_text().strip() + "\n\nCaption: " + json.dumps(row["caption"], ensure_ascii=False))
            self.assertEqual(base64.b64decode(message[0]["content"][0]["image_url"]["url"].split(",", 1)[1]), self.image_bytes)
            for private in ("SECRET_FACTCHECK", "ALTERNATIVE_CAPTION", str(self.root)):
                self.assertNotIn(private, json.dumps(payload))
        report = json.loads((self.root / "reports/verite/lfm3_vl_2b/metrics.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["invalid_responses"], 1)
        self.assertEqual(report["request_errors"], 1)
        self.assertAlmostEqual(report["majority_baseline"]["accuracy"], 4 / 6)
        for category in ("true", "miscaptioned", "out-of-context"):
            row = report["source_category_breakdown"][category]
            self.assertEqual((row["n_expected"], row["n_evaluated"], row["binary_accuracy"]), (2, 2, 0.5))
        self.assertEqual(report["source_category_breakdown"]["miscaptioned"]["prediction_counts"]["INVALID"], 1)
        self.assertEqual(report["source_category_breakdown"]["out-of-context"]["prediction_counts"]["ERROR"], 1)
        self.assertFalse((self.root / "reports/newsclippings/lfm3_vl_2b").exists())
        self.csv.rename(self.csv.with_suffix(".unavailable"))
        self.images.rename(self.images.with_name("images_unavailable"))
        with patch.object(b, "checked_manifest", side_effect=AssertionError("No raw data for completed run")), \
             patch.object(b.LocalClient, "request", side_effect=AssertionError("No HTTP")) as http:
            self.assertEqual(b.run_baseline(directory, self.config, True, self.root), 0)
            self.assertEqual(b.inspect_run(directory, self.config, self.root)["counts"]["remaining"], 0)
            self.assertEqual(build_report(directory, self.config, self.root)["accuracy"], 0.5)
            compare_results(self.root, dataset="verite")
            self.assertEqual(http.call_count, 0)
        for path in (self.root / "reports/verite").rglob("*.json"):
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values(): yield from strings(item)
                elif isinstance(value, list):
                    for item in value: yield from strings(item)
            self.assertFalse(any(str(self.root) in value for value in strings(json.loads(path.read_text()))))
            self.assertNotIn("SECRET_FACTCHECK", path.read_text())
            for row in self.rows:
                self.assertFalse(any(row["caption"] == value for value in strings(json.loads(path.read_text()))))

    def test_dataset_selection_isolation_and_changed_inputs(self):
        prepare(self.root, self.config)
        with patch.object(b, "ROOT", self.root), patch.object(b, "inspect_run", return_value={}) as inspect:
            self.assertEqual(b.main(["--dataset", "verite", "--model", "qwen35_2b", "status"]), 0)
            self.assertEqual(inspect.call_args.args[0].name, "qwen35_2b")
            self.assertEqual(inspect.call_args.args[0].parent.name, "verite")
        with self.assertRaisesRegex(ValueError, "dataset.*disagree"):
            b.load_model_config(self.root, path="configs/verite/lfm3_vl_2b.json", dataset="newsclippings")
        foreign = self.root / "foreign";foreign.mkdir()
        (foreign / "run.json").write_text(json.dumps({"kind": "newsclippings_full_test_baseline"}))
        with self.assertRaisesRegex(ValueError, "not a baseline"):
            b.run_baseline(foreign, self.config, True, self.root)
        self.rows[0]["caption"] = "Changed real input"
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Prepared VERITE inputs"):
            b.checked_manifest(self.config, self.root)

    def test_local_image_decode_failure_is_saved_as_error(self):
        bad = self.images / "images/bad.png";bad.write_bytes(b"not an image")
        self.rows[0]["image_path"] = "images/bad.png";self.write_csv()
        manifest = prepare(self.root, self.config)
        client = FakeClient([response("MISMATCH") for _ in range(5)])
        directory = self.root / self.config["run_dir"]
        with patch.object(b, "check_server", return_value=(client, {})):
            self.assertEqual(b.run_baseline(directory, self.config, root=self.root), 0)
        records, _ = b.load_journal(directory / "predictions.jsonl", manifest["records"], repair=False)
        self.assertEqual(records[0]["request_error_kind"], "media_decode")
        self.assertEqual(records[0]["status"], "request_error")
        self.assertIn("Unsupported image signature", records[0]["error"])
        self.assertEqual(len(client.calls), 5)


if __name__ == "__main__":
    unittest.main()
