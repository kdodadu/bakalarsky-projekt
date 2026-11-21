"""Uniform, offline reports. All scores come from evaluation.evaluate via inspect_run."""

import csv
import hashlib
import io
import sys
from pathlib import Path

from .baseline import MODEL_ALIASES, fingerprint, freeze_protocol, inspect_run, load_model_config, read_json
from .data import local_path, sha256_file
from .storage import write_json, write_text

SCHEMA_VERSION = 1


def protocol_summary(metadata):
    """Describe recorded evidence only; absent historical identity stays unknown."""
    server = metadata.get("server") or {}
    props = server.get("props", {}).get("response") or {}
    contract = metadata["protocol"]["contract"]
    files = server.get("loaded_files", {})
    weights = {}
    for kind in ("gguf", "mmproj"):
        asset = files.get(kind, {})
        filename = asset.get("repository_filename", asset.get("filename"))
        if kind == "gguf" and not filename and props.get("model_path"):
            filename = props["model_path"].replace("\\", "/").rsplit("/", 1)[-1]
        weights[kind] = {"filename": filename, "sha256": asset.get("sha256")}
    return {
        "fingerprint": metadata["protocol"]["fingerprint"],
        "annotations_sha256": metadata["config"]["annotations_sha256"],
        "prompt_sha256": contract["prompt_sha256"],
        "request_text_format": contract["request_text_format"],
        "client_image_preprocessing": contract["image_preprocessing"],
        "request_generation": contract["generation_config"],
        "fixed_payload_parameters": contract["fixed_payload_parameters"],
        "client_parser_evaluation_sha256": contract["code_sha256"],
        "llama_cpp_build": props.get("build_info"),
        "quantization": props.get("model_ftype"),
        "context_size": props.get("default_generation_settings", {}).get("n_ctx"),
        "parallel": props.get("total_slots"),
        "chat_template_sha256": hashlib.sha256(props["chat_template"].encode()).hexdigest() if props.get("chat_template") else None,
        "template_source": server.get("template_source"),
        "native_image_preprocessing": server.get("native_image_preprocessing"),
        "image_processing_settings": contract.get("experiment", {}).get("image_processing"),
        "capacity_preflight": contract.get("capacity_preflight"),
        "reasoning_budget": server.get("identity", {}).get("reasoning_budget"),
        "response_reasoning_format": server.get("identity", {}).get("response_reasoning_format"),
        "server_default_generation_params": props.get("default_generation_settings", {}).get("params"),
        "weights": weights,
        "repository_revision": server.get("identity", {}).get("repository_revision"),
        "launch_command": server.get("expected_launch_command"),
        "unknown_metadata": [name for name, value in (
            ("gguf_sha256", weights["gguf"]["sha256"]),
            ("mmproj_filename", weights["mmproj"]["filename"]),
            ("mmproj_sha256", weights["mmproj"]["sha256"]),
            ("observed_launch_command", server.get("observed_process", {}).get("command_line")),
            ("verified_repository_revision", server.get("identity", {}).get("repository_revision")),
            ("native_image_preprocessing", server.get("native_image_preprocessing")),
        ) if value is None],
    }


def build_report(directory, config, root):
    """One formatter for every model; journal uniqueness/parser checks are shared."""
    inspected = inspect_run(directory, config, root)
    metrics = inspected["metrics"]
    started = (directory / "run.json").is_file()
    metadata = read_json(directory / "run.json") if started else None
    manifest = read_json(directory / "input_manifest.json" if started else local_path(root, config["prepared_dir"]) / "manifest.json")
    if config["split"] != "test" or metrics["n_expected"] != 7264:
        raise ValueError("Reports require the official 7264-record merged_balanced test.")
    canonical = [{key: row[key] for key in ("annotation_index", "id", "image_id", "caption", "image_sha256", "true_label")}
                 for row in manifest["records"]]
    if len(canonical) != 7264 or {row["annotation_index"] for row in canonical} != set(range(7264)):
        raise ValueError("Test input manifest is incomplete or contains duplicate indices.")
    historical = None
    if started and (directory / "metrics.json").is_file():
        try:
            historical = read_json(directory / "metrics.json")
        except (OSError, ValueError):
            print("WARNING: stored metrics snapshot unavailable; rebuilding scores from the response journal.", file=sys.stderr)
    differences = {key: {"historical": historical.get(key), "recomputed": value}
                   for key, value in metrics.items() if historical is not None and historical.get(key) != value}
    # Historical runs have no dedicated run ID. Derive a stable ID from the
    # recorded creation timestamp and model; it does not change on relocation.
    return {
        "schema_version": SCHEMA_VERSION,
        "model": config["model_id"], "model_alias": metadata["config"]["model"] if started else config["model"],
        "source_run_id": f"{config['model_id']}@{metadata['created_at_utc']}" if started else None,
        "source_run_directory": directory.relative_to(root).as_posix() if directory.is_relative_to(root) else "<external>/" + directory.name,
        "source_run_created_at_utc": metadata["created_at_utc"] if started else None,
        "source_predictions_sha256": sha256_file(directory / "predictions.jsonl") if started else None,
        "dataset": "NewsCLIPpings", "variant": "merged_balanced", "split": "test",
        "status": "complete" if metrics["complete"] else inspected["status"],
        **metrics,
        "input_composition_sha256": fingerprint(canonical),
        "protocol": protocol_summary(metadata) if started else None,
        "planned_protocol": None if started else freeze_protocol(config, root),
        "historical_metrics_match": not differences if historical is not None else None,
        "historical_metric_differences": differences,
    }


def write_report(directory, config, root, output_dir=None):
    report = build_report(directory, config, root)
    write_json((Path(output_dir) if output_dir else root / "reports") / config["model_id"] / "metrics.json", report)
    return report


def compare_results(root, output_dir=None, data_paths=None):
    """Generate the same per-model report and one compact comparison, offline."""
    reports = []
    destination = Path(output_dir) if output_dir else root / "reports"
    for model_id in MODEL_ALIASES:
        config = load_model_config(root, model_id=model_id)
        config.update(data_paths or {})
        reports.append(build_report(local_path(root, config["run_dir"]), config, root))
    if len({r["input_composition_sha256"] for r in reports}) != 1:
        raise ValueError("Comparison input captions, image hashes, labels or annotation indices differ.")
    summaries = {r["model"]: r["protocol"] for r in reports if r["protocol"] and r["protocol"]["llama_cpp_build"]}
    shared = ("annotations_sha256", "prompt_sha256", "request_text_format", "client_image_preprocessing",
              "fixed_payload_parameters", "client_parser_evaluation_sha256")
    comparison = {"same_input_records": True,
                  "input_scope": "Saved inputs for started runs; verified prepared inputs for pending runs.",
                  "observed_models": list(summaries),
                  "pending_models": [r["model"] for r in reports if r["status"] == "pending"],
                  "shared_checks_scope": "Only runs with recorded server observations; planned settings are not observations.",
                  **{f"same_{key}": len({fingerprint(p[key]) for p in summaries.values()}) == 1 if len(summaries) > 1 else None for key in shared},
                  "same_request_generation_except_model": (len({fingerprint({k: v for k, v in p["request_generation"].items() if k != "model"})
                                                                for p in summaries.values()}) == 1) if len(summaries) > 1 else None,
                  "different_or_unknown": {key: {r["model"]: r["protocol"][key] if r["protocol"] else None for r in reports}
                                           for key in ("llama_cpp_build", "context_size", "parallel", "chat_template_sha256", "weights",
                                                       "server_default_generation_params", "reasoning_budget", "response_reasoning_format", "unknown_metadata")}}
    columns = ("model", "status", "n_evaluated", "n_expected", "accuracy", "macro_f1", "valid_response_rate", "invalid_responses", "request_errors", "complete")
    rows = [{key: report[key] for key in columns} for report in reports]
    for report in reports:
        write_json(destination / report["model"] / "metrics.json", report)
    write_json(destination / "model_comparison.json", {"schema_version": SCHEMA_VERSION, "models": rows,
               "protocol_comparison": comparison, "policy": reports[0]["policy"]})
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    write_text(destination / "model_comparison.csv", text.getvalue())
    for row in rows:
        accuracy = "unknown" if row["accuracy"] is None else f"{row['accuracy']:.6f}"
        f1 = "unknown" if row["macro_f1"] is None else f"{row['macro_f1']:.6f}"
        print(f"{row['model']:20} {row['status']:10} {row['n_evaluated']}/{row['n_expected']} accuracy={accuracy} macro-F1={f1} complete={row['complete']}")
    return reports


def refresh_after_run(directory, config, root, output_dir=None):
    """Derived failures never affect the authoritative response journal/resume."""
    if config.get("split") != "test" or config.get("expected_records") != 7264:
        return
    try:
        # Only the configured run is eligible for the current comparison.
        if directory.resolve() != local_path(root, config["run_dir"]):
            print("Reports skipped for an unselected run-dir; use report --run-dir explicitly.", file=sys.stderr)
            return
        write_report(directory, config, root, output_dir)
        compare_results(root, output_dir, {key: config[key] for key in ("annotations", "archive", "prepared_dir") if key in config})
    except Exception as exc:
        print(f"WARNING: derived reports could not be refreshed: {type(exc).__name__}: {exc}. "
              "Saved responses remain authoritative; resume will skip them. "
              "Recover without inference: .\\.venv\\Scripts\\python.exe scripts/baseline.py compare", file=sys.stderr)
