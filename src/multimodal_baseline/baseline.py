"""Full test orchestration; inference, parsing and metrics reuse the pilot unchanged."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

from .client import LocalClient, interpret, make_payload
from .data import dataset_name, image_candidates, input_identity, local_path, prepare, select_test_records, sha256_file, validate_verite_snapshot, verite_records
from .evaluation import evaluate
from .storage import log_event, write_json

ROOT = Path(__file__).resolve().parents[2]
MODEL_ALIASES = {
    "lfm3_vl_2b": "LiquidAI/LFM3-VL-2B-GGUF:Q4_K_M",
    "qwen35_2b": "unsloth/Qwen3.5-2B-GGUF:Q4_K_M",
    "smolvlm2_2_2b": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M",
}
FROZEN_FILES = ("src/multimodal_baseline/client.py", "src/multimodal_baseline/evaluation.py")
GENERATION_KEYS = ("seed", "base_url", "model", "temperature", "repeat_penalty",
                   "chat_template_kwargs", "max_tokens", "timeout_seconds")
SNAPSHOT_EVERY = 25
# Audited predecessor: same frozen client/parser/metrics; only orchestration is repaired.
LEGACY_RUNNER_SHA256 = "2deaf6ec23fb5c9a5cbd20bcebe86ff39b88e92b87e8f7fe33d2e9bee366dd38"
PREVIOUS_RUNNER_SHA256 = "24ff9028d964eb3bf627c70e929d4ed11b39afefee6017c446e47c65b202c177"
STANDALONE_RUNNER_SHA256 = "f817e191488b50dec17260ab86d8ebf398425167edee2c496e469b3869237b47"
MULTIMODEL_RUNNER_SHA256 = "8f422c1aa56a405a7e3ea97af138c7600cb4c6681e2cc087836c5d7b1560d810"
SAMPLE_ERROR_RUNNER_SHA256 = "9e807700f68e03254979047a1f4beeb69e3826c693819480f0c1ddb5da5e894f"
RECOVERY_RUNNER_SHA256 = "6df0d6c72fc759574f630959a649789923dd557bacf05b53b22c24575a428e85"
PENDING_RUNNER_SHA256 = "f30c6239f4b4454f84e67486ef342c9fe7ea3a4befa0457f919d5f5af79d48ac"
CAPACITY_PREDECESSOR_RUNNER_SHA256 = "90d1d5b86b2a0d6a8dec5bfd68cc7bb6a1b4c594cc1aec094cc36f1b63b06eee"
# The server/vision checks in this helper are unchanged. Only report code and
# the compatibility comparison for relocated historical runs were refactored.
PREVIOUS_HELPER_SHA256 = "1e0446c41a97dcbea3da5b018666cc52c85b97fc0aac212884a910946f3c713b"
# The failed LFM preflight used byte equality before accounting for the server's
# terminal LF removal. No classification or request settings change in this fix.
TEMPLATE_HELPER_SHA256 = "7e3bc62488a245d4e9d0ef2f197b16083a12138d3e67cd1685f3e585170ee62c"
CAPACITY_PREDECESSOR_HELPER_SHA256 = "99b1274b96c0621310ee0de051c73c54859f7c190e9dc8068a2910502b33b1c9"
# The last published runner/helper: removal of local diagnostics and automatic
# val gating does not change classification requests, parsing or scoring.
PRE_SIMPLIFICATION_RUNNER_SHA256 = "6fe2d13457baff80849d3b84337f59f31cc46ecfdb86f79f7343888b2c5b5060"
PRE_SIMPLIFICATION_HELPER_SHA256 = "6eadcde6461dbfbf8b5e5acfdf2fc07caedb096f4ce6b414e5709de752fa0887"
PRE_VERITE_RUNNER_SHA256 = "7e97f608342d6643200a1cca2f0ee908df79120b8e6e3acd7bb30d9cb67a97d2"
# Audited layout-only predecessor. Historical snapshots retain their old paths.
PRE_LAYOUT_RUNNER_SHA256 = "7dbfd49756a79b085a7e4a8719ed75cc11d78a70c4c06e5c8ce02fec5ea3368a"
# Audited extension: explicit VERITE image opt-in only. News payload and the
# strict interpret() parser are unchanged; other client revisions stay rejected.
PRE_VERITE_CLIENT_SHA256 = "4864bff73635fa635c65aeca3d3eb53787e7bd7a7caf4a67c78c365496d9bccc"
VERITE_CLIENT_SHA256 = "a50b58035d57fe4397874ef77160750f517c8f932d5d6ab669c94f73824b2cb1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_model_config(root, *, model_id=None, path=None, dataset=None):
    selected_dataset = dataset or "newsclippings"
    config = read_json(local_path(root, path or f"configs/{selected_dataset}/{model_id}.json"))
    identifier = config.get("model_id")
    if identifier not in MODEL_ALIASES or (model_id and identifier != model_id):
        raise ValueError("Select a configuration with a matching, known model_id.")
    if config.get("model") != MODEL_ALIASES[identifier]:
        raise ValueError("model_id and the expected server model disagree.")
    if dataset is not None and dataset_name(config) != dataset:
        raise ValueError("Selected dataset and configuration disagree.")
    dataset_name(config)
    return config


def protocols_match(saved, current):
    """Compare inference strictly; historical diagnostics remain metadata only."""
    for value in (saved, current):
        if "contract" not in value or value.get("fingerprint") != fingerprint(value["contract"]):
            return False
    previous = json.loads(json.dumps(saved["contract"]))
    actual = json.loads(json.dumps(current["contract"]))
    # Only module locations change. Digest values must still match the audited
    # client/evaluation revisions; never rewrite a historical snapshot on disk.
    for contract in (previous, actual):
        hashes = contract.get("code_sha256", {})
        for name in ("client.py", "evaluation.py"):
            legacy = "src/newsclip_pilot/" + name
            current_name = "src/multimodal_baseline/" + name
            if legacy in hashes:
                if current_name in hashes:
                    return False
                hashes[current_name] = hashes.pop(legacy)
    old_helper = previous.get("orchestration_sha256")
    if old_helper in (PREVIOUS_HELPER_SHA256, TEMPLATE_HELPER_SHA256, CAPACITY_PREDECESSOR_HELPER_SHA256, PRE_SIMPLIFICATION_HELPER_SHA256) and "orchestration_sha256" in actual:
        previous["orchestration_sha256"] = actual["orchestration_sha256"]
    client = "src/multimodal_baseline/client.py"
    if (previous.get("code_sha256", {}).get(client) == PRE_VERITE_CLIENT_SHA256
            and actual.get("code_sha256", {}).get(client) == VERITE_CLIENT_SHA256):
        previous["code_sha256"][client] = VERITE_CLIENT_SHA256
    from .experiments import experiment_identity
    for contract in (previous, actual):
        contract.pop("capacity_preflight", None)
        if "experiment" in contract:
            contract["experiment"] = experiment_identity(contract["experiment"])
        if "generation_config" in contract:
            contract["generation_config"].pop("base_url", None)
    return previous == actual


def config_identity(config):
    from .experiments import experiment_identity
    operational = {"reference_pilot", "protocol", "model_id", "run_dir", "validation", "capacity_preflight",
                   "annotations", "archive", "images_dir", "prepared_dir", "prompt_file", "server_log", "base_url"}
    result = {key: value for key, value in config.items() if key not in operational}
    if "experiment" in result:
        result["experiment"] = experiment_identity(result["experiment"])
    return result


def run_kind(config):
    return "verite_adapted_binary_baseline" if dataset_name(config) == "verite" else "newsclippings_full_test_baseline"


def freeze_protocol(config, root=ROOT):
    reference = config["protocol"]
    prompt = local_path(root, config["prompt_file"]).read_text(encoding="utf-8").strip()
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != reference["prompt_sha256"]:
        raise ValueError("Prompt differs from the configured protocol.")
    code_hashes = {name: sha256_file(root / name) for name in FROZEN_FILES}
    if any(code_hashes[name] != reference["code_sha256"][name] for name in FROZEN_FILES):
        raise ValueError("Inference/parser or evaluation code differs from the configured protocol.")
    contract = {"generation_config": {key: config[key] for key in GENERATION_KEYS},
                "prompt": prompt, "prompt_sha256": reference["prompt_sha256"], "code_sha256": code_hashes,
                "request_text_format": "prompt + newline newline + Caption: + JSON-quoted caption",
                "image_preprocessing": "original bytes; no client resizing or recompression",
                "fixed_payload_parameters": {"presence_penalty": 0.0, "frequency_penalty": 0.0,
                                             "stream": False, "cache_prompt": False}}
    if "experiment" in config:
        contract["experiment"] = config["experiment"]
        contract["orchestration_sha256"] = sha256_file(Path(__file__).with_name("experiments.py"))
    if dataset_name(config) == "verite":
        from .data import VERITE_BINARY_MAPPING
        if config.get("binary_label_mapping") != VERITE_BINARY_MAPPING:
            raise ValueError("VERITE adapted binary label mapping differs from the approved protocol.")
        contract["dataset_protocol"] = {"dataset": "VERITE", "variant": config["variant"],
                                        "evaluation_protocol": "adapted binary evaluation",
                                        "binary_label_mapping": config["binary_label_mapping"]}
    return {"contract": contract, "fingerprint": fingerprint(contract)}


def checked_manifest(config, root=ROOT):
    manifest_path = local_path(root, config["prepared_dir"]) / "manifest.json"
    manifest = read_json(manifest_path)
    if dataset_name(config) == "verite":
        current = verite_records(root, config)
        if input_identity(manifest) != input_identity(current):
            raise ValueError("Prepared VERITE inputs/categories/mapping changed; rerun prepare for a new experiment.")
        return manifest_path, current
    selected = select_test_records(local_path(root, config["annotations"]), seed=config["seed"])
    if selected["annotations_sha256"] != config["annotations_sha256"]:
        raise ValueError("test.json differs from the verified official merged_balanced file.")
    for key in ("split", "seed", "pair_count", "annotations_sha256", "selected_pair_indices", "class_counts"):
        if manifest.get(key) != selected[key]:
            raise ValueError(f"Prepared test manifest mismatch: {key}.")
    if len(manifest["records"]) != config["expected_records"] or config["expected_records"] != 7264:
        raise ValueError("Full merged_balanced test requires exactly 7264 records.")
    subset = read_json(local_path(root, config["prepared_dir"]) / "visualnews_subset.json")
    visualnews = {r["id"]: r for r in subset["records"]}
    if subset["archive"] != manifest["archive"] or subset["metadata_member"] != manifest["metadata_member"]:
        raise ValueError("VisualNews metadata provenance differs from manifest.")
    checked_images = set()
    for actual, expected in zip(manifest["records"], selected["records"], strict=True):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Test annotation mismatch at {expected['annotation_index']}.")
        if actual["caption"] != visualnews[actual["id"]]["caption"]:
            raise ValueError("Caption is not VisualNews[id].caption.")
        if actual["archive_member"] not in image_candidates(manifest["metadata_member"]["name"], visualnews[actual["image_id"]]["image_path"]):
            raise ValueError("Image path is not VisualNews[image_id].image_path.")
        image = local_path(root, actual["image_path"])
        image_root = local_path(root, config["prepared_dir"]) / "images"
        if not image.is_relative_to(image_root) or not image.is_file():
            # Prepared folders may move independently of the project. Only the
            # same basename inside the selected prepared folder is eligible;
            # its recorded content hash is still checked below.
            image = (image_root / Path(actual["image_path"]).name).resolve()
        if not image.is_relative_to(image_root):
            raise ValueError("Image is outside prepared test images.")
        actual["image_path"] = image.relative_to(root).as_posix() if image.is_relative_to(root) else str(image)
        if image not in checked_images:
            if sha256_file(image) != actual["image_sha256"]:
                raise ValueError(f"Missing or changed prepared image: {image}")
            checked_images.add(image)
    return manifest_path, manifest


@contextmanager
def exclusive_run(directory):
    """OS-owned lock is released even after a crash; the small lock file may remain."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".run.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another process is using this run directory.") from None
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def initialize(directory, config, protocol, manifest_path, manifest, root=ROOT):
    if (directory / "run.json").exists():
        raise FileExistsError("Run is already initialized; use run or run --resume.")
    if any(p.name != ".run.lock" for p in directory.iterdir()):
        raise ValueError("New run directory must be empty.")
    metadata = {"schema_version": 1, "kind": run_kind(config), "split": config.get("split", "test"),
                "created_at_utc": now(), "status": "ready", "complete": False,
                "n_completed": 0, "expected_records": len(manifest["records"]),
                "config": config, "protocol": protocol, "prompt": protocol["contract"]["prompt"],
                "manifest_sha256": sha256_file(manifest_path), "input_manifest_sha256": fingerprint(manifest),
                "input_identity_sha256": fingerprint(input_identity(manifest)),
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "runner_sha256": sha256_file(Path(__file__)),
                "generation_order": "CSV source order and stable source IDs; no sampling or shuffle" if dataset_name(config) == "verite" else "official annotation order, no sampling or shuffle",
                "sessions": [], "recoveries": [], "server": None}
    if dataset_name(config) == "verite":
        metadata["source_annotations_sha256"] = manifest["annotations_sha256"]
    write_json(directory / "input_manifest.json", manifest)
    (directory / "predictions.jsonl").touch(exist_ok=False)
    write_json(directory / "metrics.json", evaluate([], len(manifest["records"])))
    write_json(directory / "run.json", metadata)
    return metadata


def load_journal(path, manifest_rows, *, repair=True):
    """Recover only an incomplete final write; never drop valid responses or duplicates."""
    expected = {r["annotation_index"]: r for r in manifest_rows}
    records, seen, recovery = [], set(), None
    with path.open("r+b" if repair else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                result = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError(f"Corrupt complete JSONL line at byte {offset}; refusing to discard it.") from None
                if not repair:
                    raise ValueError("Incomplete final JSONL write; run --resume will preserve and repair its tail.") from None
                backup = path.parent / f"incomplete_tail_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.bin"
                backup.write_bytes(line)
                handle.seek(offset)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                recovery = {"at_utc": now(), "action": "preserved_and_removed_incomplete_final_write", "backup": backup.name, "bytes": len(line)}
                break
            index = result["annotation_index"]
            if index in seen or index not in expected:
                raise ValueError(f"Duplicate or unexpected saved annotation_index={index}.")
            for key in ("id", "image_id", "pair_index", "true_label", "source_category"):
                if key in expected[index] and result.get(key) != expected[index][key]:
                    raise ValueError(f"Saved result does not match manifest: {index}, {key}.")
            parsed = interpret(result)
            if any(result[key] != parsed[key] for key in parsed):
                raise ValueError(f"Saved parsing differs from the frozen pilot parser: {index}.")
            records.append(result)
            seen.add(index)
            if not line.endswith(b"\n"):
                if not repair:
                    break
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                recovery = {"at_utc": now(), "action": "added_missing_final_newline", "annotation_index": index}
                break
    evaluate(records, len(manifest_rows))
    return records, recovery


def append_result(handle, result):
    data = (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())


def save_progress(directory, metadata, records):
    metrics = evaluate(records, metadata["expected_records"])
    metadata.update(n_completed=len(records), complete=metrics["complete"], updated_at_utc=now())
    # These are derived snapshots. A Windows reader must not stop the durable journal.
    deferred = metadata.setdefault("deferred_snapshots", [])
    for name, value in (("metrics.json", metrics), ("run.json", metadata)):
        was_deferred = name in deferred
        if was_deferred:
            deferred.remove(name)
        try:
            write_json(directory / name, value)
            if was_deferred:
                log_event(directory, "snapshot_recovered", file=name, saved=len(records))
        except PermissionError as exc:
            if name not in deferred:
                deferred.append(name)
            if not was_deferred:
                log_event(directory, "snapshot_deferred", file=name, saved=len(records), error=str(exc))
                print(f"WARNING: {name} is locked; answers are saved. Snapshot update deferred.", file=sys.stderr)
    return metrics


class StopRequest:
    def __init__(self):
        self.requested = False

    def handle_signal(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("Ctrl+C: will save the current response, then stop. A second Ctrl+C aborts the in-flight request.", flush=True)


def request_error_kind(result):
    """Only known sample-specific HTTP failures permit automatic continuation."""
    if result.get("http_status") != 400:
        if str(result.get("error") or "").startswith(("URLError:", "TimeoutError:", "ConnectionResetError:",
                                                       "ConnectionAbortedError:", "ConnectionRefusedError:",
                                                       "RemoteDisconnected:", "OSError:")):
            return "transport"
        return "infrastructure_or_unknown"
    try:
        error = json.loads(result.get("raw_http_response") or "{}")["error"]
        if error.get("type") == "exceed_context_size_error":
            return "context_limit"
        if error.get("message") == "Failed to load image or audio file":
            return "media_decode"
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return "infrastructure_or_unknown"


def recover_connection(directory, metadata, stop, root=ROOT, config=None):
    """Probe health only, then revalidate identity. Never retry a classification."""
    config = config or metadata["config"]
    info = {"annotation_index": metadata["current_annotation_index"], "health_attempts": 0,
            "started_at_utc": now(), "recovered": False}
    metadata["last_connection_recovery"] = info
    metadata["phase"] = "connection_recovery"
    log_event(directory, "connection_recovery_started", **info)
    if not metadata.get("server"):
        info["error"] = "No previously verified server snapshot; automatic recovery refused."
        return None
    health_client = LocalClient(config["base_url"], timeout=5)
    for delay in (0, 1, 2, 4):
        if stop.requested:
            break
        if delay:
            time.sleep(delay)
        if stop.requested:
            break
        info["health_attempts"] += 1
        health = health_client.request("/health")
        log_event(directory, "connection_health_check", annotation_index=info["annotation_index"],
                  attempt=info["health_attempts"], http_status=health.get("http_status"), error=health.get("error"))
        if health.get("error") or not isinstance(health.get("response"), dict) or health["response"].get("status") != "ok":
            info["error"] = health.get("error") or str(health.get("response"))
            continue
        if stop.requested:
            break
        try:
            # Short GET timeouts during preflight; original classification timeout
            # is restored before returning the newly verified client.
            client, server = check_server({**config, "timeout_seconds": 5}, root, metadata["server"])
            client.timeout = config["timeout_seconds"]
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            info["error"] = f"Server revalidation failed: {exc}"
            log_event(directory, "connection_revalidation_failed", **info)
            return None
        info.update(recovered=True, finished_at_utc=now())
        info.pop("error", None)
        log_event(directory, "connection_recovered", **info)
        return client, server
    info["finished_at_utc"] = now()
    log_event(directory, "connection_recovery_failed", **info)
    return None


def process_pending(directory, metadata, rows, records, client, stop, root=ROOT, config=None):
    indices = {row["annotation_index"] for row in rows}
    if len(indices) != len(rows) or len(rows) != metadata["expected_records"]:
        raise ValueError("Queue must contain every expected unique test annotation.")
    done = {r["annotation_index"] for r in records}
    if len(done) != len(records) or not done <= indices:
        raise ValueError("Saved records are duplicated or absent from the queue.")
    config = config or metadata["config"]
    prompt = metadata["prompt"]
    metadata["status"] = "running"
    metadata["stop_reason"] = None
    save_progress(directory, metadata, records)
    with (directory / "predictions.jsonl").open("ab") as handle:
        for row in rows:
            if stop.requested:
                metadata["status"] = "interrupted"
                metadata["stop_reason"] = {"kind": "ctrl_c", "phase": "between_records"}
                break
            if row["annotation_index"] in done:
                continue
            metadata["current_annotation_index"] = row["annotation_index"]
            metadata["phase"] = "build_payload"
            text = prompt + "\n\nCaption: " + json.dumps(row["caption"], ensure_ascii=False)
            started_at = now()
            local_media_error = False
            try:
                payload = make_payload(config, local_path(root, row["image_path"]), text)
            except (OSError, ValueError) as exc:
                local_media_error = True
                response = {"response": None, "raw_http_response": None, "http_status": None,
                            "error": f"InputError: {type(exc).__name__}: {exc}", "elapsed_seconds": 0.0}
            else:
                metadata["phase"] = "request"
                log_event(directory, "request_started", annotation_index=row["annotation_index"], saved=len(records))
                response = client.request("/v1/chat/completions", payload)
            result = {**{key: row[key] for key in ("annotation_index", "pair_index", "id", "image_id", "true_label", "source_category") if key in row},
                      "started_at_utc": started_at, **response, **interpret(response)}
            if result["status"] == "request_error":
                result["request_error_kind"] = "media_decode" if local_media_error else request_error_kind(result)
            metadata["phase"] = "save_response"
            append_result(handle, result)
            records.append(result)
            done.add(row["annotation_index"])
            log_event(directory, "response_saved", annotation_index=row["annotation_index"], saved=len(records), status=result["status"])
            metadata["phase"] = "update_snapshots"
            if len(records) % SNAPSHOT_EVERY == 0:
                save_progress(directory, metadata, records)
            print(f"[{len(records)}/{len(rows)}] annotation={row['annotation_index']} {result['status']} {result['prediction']} ({result['elapsed_seconds']:.2f}s)", flush=True)
            if result["status"] == "request_error":
                details = {"annotation_index": row["annotation_index"], "http_status": result["http_status"],
                           "error_kind": result["request_error_kind"], "raw_http_response": result["raw_http_response"]}
                if result["request_error_kind"] in ("context_limit", "media_decode"):
                    log_event(directory, "sample_error_continue", **details)
                    print(f"Sample-specific {result['request_error_kind']} saved as ERROR; continuing without retry or preprocessing changes.", flush=True)
                    continue
                if result["request_error_kind"] == "transport":
                    print(f"Connection failure saved as ERROR: {result['error']}. Checking server health without retrying this example.", flush=True)
                    recovered = recover_connection(directory, metadata, stop, root, config=config)
                    if recovered is not None:
                        client, metadata["server"] = recovered
                        print("Same model/protocol verified after connection recovery; continuing with the next example.", flush=True)
                        continue
                metadata["status"] = "stopped_on_request_error"
                metadata["stop_reason"] = {"kind": "request_error", "annotation_index": row["annotation_index"],
                                           "http_status": result["http_status"], "error": result["error"]}
                if result["request_error_kind"] == "transport":
                    metadata["stop_reason"]["connection_recovery"] = metadata["last_connection_recovery"]
                    if stop.requested:
                        metadata["status"] = "interrupted"
                        metadata["stop_reason"]["kind"] = "ctrl_c"
                log_event(directory, "request_error_stop", **details)
                print("Infrastructure/unknown request error saved and counted; stopping. Resume skips this saved record.", flush=True)
                break
    if len(records) == metadata["expected_records"]:
        metadata["status"] = "completed"
        metadata["stop_reason"] = {"kind": "all_expected_records_saved"}
    return save_progress(directory, metadata, records)


def server_signature(props):
    return {"model_alias": props.get("model_alias"), "model_ftype": props.get("model_ftype"),
            "model_filename": str(props.get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1],
            "build_info": props.get("build_info"),
            "chat_template_sha256": hashlib.sha256(props.get("chat_template", "").encode("utf-8")).hexdigest(),
            "default_generation_settings": props.get("default_generation_settings")}


def check_server(config, root=ROOT, saved_server=None):
    client = LocalClient(config["base_url"], config["timeout_seconds"])
    server = client.probe(config["model"])
    if "experiment" in config:
        from .experiments import verify_model_server
        return client, verify_model_server(config, server, root, saved_server)
    expected = (server_signature(saved_server["props"]["response"]) if saved_server
                else config["protocol"]["server"])
    new_props = server["props"]["response"]
    if not isinstance(new_props, dict):
        raise RuntimeError("Cannot verify the local model/template/build using /props.")
    actual = server_signature(new_props)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"Server {key} differs from the frozen protocol; baseline was not started.")
    return client, server


def verify_run(directory, config, protocol, manifest_path=None, manifest=None):
    metadata = read_json(directory / "run.json")
    expected_kind = run_kind(config)
    if metadata.get("kind") != expected_kind:
        raise ValueError("This is not a baseline run; pilot directories are read-only.")
    # These describe storage, a compatibility gate or redundant identification,
    # not classification. Actual model, input manifests and protocol stay strict.
    saved_protocol = metadata["protocol"]
    if (config_identity(metadata["config"]) != config_identity(config)
            or (metadata["config"].get("model_id") is not None
                and metadata["config"]["model_id"] != config.get("model_id"))
            or not protocols_match(saved_protocol, protocol)
            or metadata["prompt"] != protocol["contract"]["prompt"]):
        raise ValueError("Settings differ from this run's saved snapshot. Resume refused.")
    if "experiment" not in config and metadata.get("server") and config.get("protocol", {}).get("server") != server_signature(metadata["server"]["props"]["response"]):
        raise ValueError("Configured server protocol differs from the saved run.")
    current_runner = sha256_file(Path(__file__))
    previous_runner = metadata.get("active_runner_sha256", metadata["runner_sha256"])
    if previous_runner not in (current_runner, LEGACY_RUNNER_SHA256, PREVIOUS_RUNNER_SHA256, STANDALONE_RUNNER_SHA256, MULTIMODEL_RUNNER_SHA256, SAMPLE_ERROR_RUNNER_SHA256, RECOVERY_RUNNER_SHA256, PENDING_RUNNER_SHA256, CAPACITY_PREDECESSOR_RUNNER_SHA256, PRE_SIMPLIFICATION_RUNNER_SHA256, PRE_VERITE_RUNNER_SHA256, PRE_LAYOUT_RUNNER_SHA256):
        raise ValueError("Unknown runner revision; automatic compatibility has not been verified.")
    saved_manifest = read_json(directory / "input_manifest.json")
    saved_identity = fingerprint(input_identity(saved_manifest))
    if (metadata["input_manifest_sha256"] != fingerprint(saved_manifest)
            or metadata.get("input_identity_sha256", saved_identity) != saved_identity
            or (manifest is not None and fingerprint(input_identity(manifest)) != saved_identity)):
        raise ValueError("Run or prepared input manifest changed; resume refused.")
    if (metadata["expected_records"] != len(saved_manifest["records"])
            or len({r["annotation_index"] for r in saved_manifest["records"]}) != len(saved_manifest["records"])):
        raise ValueError("Expected record count differs from the verified manifest.")
    if (config.get("annotations_sha256") is not None
            and saved_manifest.get("annotations_sha256") != config["annotations_sha256"]):
        raise ValueError("Saved input belongs to another annotation file.")
    if dataset_name(config) == "verite":
        validate_verite_snapshot(config, saved_manifest)
    return metadata, previous_runner, current_runner


def display_metrics(records, expected_count):
    """Keep the frozen score policy; unattempted performance is undefined."""
    metrics = evaluate(records, expected_count)
    if not records:
        metrics["per_class"] = {label: {"precision": None, "recall": None, "f1": None, "support": 0}
                                for label in metrics["per_class"]}
        for name in ("confusion_matrix", "confusion_matrix_with_failures"):
            metrics[name] = {**metrics[name], "values": None}
    return metrics


def inspect_run(directory, config, root=ROOT):
    """Read-only end-to-end validation, with counts computed from the journal."""
    protocol = freeze_protocol(config, root)
    if (directory / "run.json").is_file():
        metadata, old_runner, current_runner = verify_run(directory, config, protocol)
        manifest = read_json(directory / "input_manifest.json")
        records, _ = load_journal(directory / "predictions.jsonl", manifest["records"], repair=False)
    else:
        if directory.exists() and (not directory.is_dir() or any(p.name != ".run.lock" for p in directory.iterdir())):
            raise ValueError("Directory has files but no run.json; refusing to treat unknown answers as a pending run.")
        if dataset_name(config) == "verite":
            manifest = read_json(local_path(root, config["prepared_dir"]) / "manifest.json")
            validate_verite_snapshot(config, manifest)
        else:
            _, manifest = checked_manifest(config, root)
        metadata = {"status": "pending", "sessions": [], "expected_records": len(manifest["records"])}
        records = []
        old_runner = current_runner = sha256_file(Path(__file__))
    done = {row["annotation_index"] for row in records}
    pending = [row["annotation_index"] for row in manifest["records"] if row["annotation_index"] not in done]
    sessions = metadata.get("sessions", [])
    metrics = display_metrics(records, metadata["expected_records"])
    return {"directory": str(directory), "stored_status": metadata["status"],
            "status": "completed" if metrics["complete"] else "incomplete" if records else "pending",
            "counts": {"raw_test": manifest.get("audit", {}).get("records", len(manifest["records"])), "prepared_selection": len(manifest["records"]),
                       "prepared_manifest": len(manifest["records"]), "run_queue": len(manifest["records"]),
                       "saved_unique_records": len(done), "remaining": len(pending),
                       "previously_saved_at_last_session_start": sessions[-1]["already_saved"] if sessions else 0},
            "next_annotation_index": pending[0] if pending else None,
            "stop_reason": metadata.get("stop_reason") or (sessions[-1].get("error") if sessions else None),
            "compatible_runner_upgrade": old_runner != current_runner,
            "frozen_protocol_verified": True, "images_verified": not (directory / "run.json").is_file() and dataset_name(config) != "verite",
            "input_check": "saved/prepared snapshot and journal; no external data read" if (directory / "run.json").is_file() or dataset_name(config) == "verite" else "prepared image bytes and annotations",
            "metrics": metrics}


def run_baseline(directory, config, resume=False, root=ROOT, stop_request=None, output_dir=None):
    if resume and not (directory / "run.json").is_file():
        raise FileNotFoundError("Cannot resume a run without run.json; start it with run first.")
    if (directory / "run.json").exists() and read_json(directory / "run.json").get("kind") != run_kind(config):
        raise ValueError("This directory is not a baseline run; it will not be modified.")
    protocol = freeze_protocol(config, root)
    if resume and (directory / "run.json").exists():
        previous, _, _ = verify_run(directory, config, protocol)
        saved_manifest = read_json(directory / "input_manifest.json")
        existing, _ = load_journal(directory / "predictions.jsonl", saved_manifest["records"], repair=False)
        if len(existing) == previous["expected_records"]:
            print(json.dumps(evaluate(existing, previous["expected_records"]), indent=2))
            from .reports import refresh_after_run
            refresh_after_run(directory, config, root, output_dir)
            print("Already complete; no server requests or run-file writes made. Derived reports refreshed if available.")
            return 0
    manifest_path, manifest = checked_manifest(config, root)
    with exclusive_run(directory):
        if not (directory / "run.json").exists():
            if resume:
                raise FileNotFoundError("Cannot resume a run without run.json.")
            initialize(directory, config, protocol, manifest_path, manifest, root)
        metadata, old_runner, current_runner = verify_run(directory, config, protocol, manifest_path, manifest)
        journal = directory / "predictions.jsonl"
        if journal.stat().st_size and not resume:
            raise ValueError("Results already exist. Use --resume to skip saved records without duplication.")
        records, recovery = load_journal(journal, manifest["records"])
        if recovery:
            metadata["recoveries"].append(recovery)
            log_event(directory, "journal_recovered", **recovery)
        if old_runner != current_runner:
            revision = {"at_utc": now(), "from_sha256": old_runner, "to_sha256": current_runner,
                        "reason": "Audited layout/module migration; content identity, inference requests, strict parser and scoring unchanged."}
            metadata.setdefault("runner_revisions", []).append(revision)
            metadata["active_runner_sha256"] = current_runner
            log_event(directory, "compatible_runner_upgrade", **revision)
        metrics = save_progress(directory, metadata, records)
        if metrics["complete"]:
            metadata["status"] = "completed"
            save_progress(directory, metadata, records)
            from .reports import refresh_after_run
            refresh_after_run(directory, config, root, output_dir)
            print(f"Already complete: {len(records)}/{metadata['expected_records']}; no server requests made.")
            return 0
        session = {"started_at_utc": now(), "resumed": resume, "already_saved": len(records),
                   "queued": metadata["expected_records"] - len(records), "runner_sha256": current_runner}
        metadata["sessions"].append(session)
        metadata["status"] = "starting"
        metadata["stop_reason"] = None
        metadata["phase"] = "server_preflight"
        log_event(directory, "session_started", **session)
        stop = stop_request or StopRequest()
        previous_handler = signal.signal(signal.SIGINT, stop.handle_signal)
        try:
            client, metadata["server"] = check_server(config, root, metadata.get("server"))
            print(f"Pending: {len(manifest['records']) - len(records)} / {metadata['expected_records']}. Directory: {directory}", flush=True)
            process_pending(directory, metadata, manifest["records"], records, client, stop, root, config=config)
        except KeyboardInterrupt:
            metadata["status"] = "interrupted"
            metadata["stop_reason"] = {"kind": "forced_ctrl_c", "phase": metadata.get("phase"),
                                       "annotation_index": metadata.get("current_annotation_index")}
        except Exception as exc:
            metadata["status"] = "failed"
            session["error"] = f"{type(exc).__name__}: {exc}"
            metadata["stop_reason"] = {"kind": "exception", "phase": metadata.get("phase"),
                                       "annotation_index": metadata.get("current_annotation_index"),
                                       "error": session["error"], "traceback": traceback.format_exc()}
            log_event(directory, "exception", **metadata["stop_reason"])
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            # Disk journal is authoritative, including a response committed before an interrupt.
            records, recovery = load_journal(journal, manifest["records"])
            if recovery:
                metadata["recoveries"].append(recovery)
            session.update(finished_at_utc=now(), saved_after=len(records))
            if len(records) == metadata["expected_records"]:
                metadata["status"] = "completed"
                metadata["stop_reason"] = {"kind": "all_expected_records_saved"}
            session["stop_reason"] = metadata.get("stop_reason")
            log_event(directory, "session_stopped", status=metadata["status"], saved=len(records),
                      remaining=metadata["expected_records"] - len(records), reason=metadata.get("stop_reason"))
            metrics = save_progress(directory, metadata, records)
            print(json.dumps({"directory": str(directory), "status": metadata["status"], **metrics}, indent=2))
            from .reports import refresh_after_run
            refresh_after_run(directory, config, root, output_dir)
        return 0 if metrics["complete"] else 130 if metadata["status"] == "interrupted" else 1


def smoke(config, image, caption, root=ROOT):
    """One explicit user input, printed only; never part of baseline metrics."""
    if not caption.strip():
        raise ValueError("smoke requires a nonempty caption.")
    protocol = freeze_protocol(config, root)
    text = protocol["contract"]["prompt"] + "\n\nCaption: " + json.dumps(caption, ensure_ascii=False)
    payload = make_payload(config, local_path(root, image), text)
    client, _ = check_server(config, root)
    result = client.request("/v1/chat/completions", payload)
    parsed = interpret(result)
    print(json.dumps({"kind": "optional_smoke_not_scored", **result, **parsed}, ensure_ascii=False, indent=2))
    return 0 if parsed["status"] == "ok" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared local binary baseline on NewsCLIPpings or adapted VERITE; no training.")
    parser.add_argument("--dataset", choices=("newsclippings", "verite"), help="Default: NewsCLIPpings, or infer from --config.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=tuple(MODEL_ALIASES))
    selection.add_argument("--config", help="Explicit configs/<dataset>/<model_id>.json; no default model.")
    for flag in ("annotations", "archive", "images-dir", "prepared-dir", "base-url", "server-log"):
        parser.add_argument("--" + flag, help="Local path/endpoint override; inference identity is preserved.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Prepare the selected complete dataset without inference.")
    run = commands.add_parser("run", help="Start or resume the complete test baseline.")
    run.add_argument("--run-dir", default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--output-dir", help="Optional destination for derived reports; run files stay in run-dir.")
    status = commands.add_parser("status", help="Verify data, queue and journal; report progress without inference or writes.")
    status.add_argument("--run-dir", default=None)
    report = commands.add_parser("report", help="Recompute the selected dataset/model report from saved answers; no HTTP or run writes.")
    report.add_argument("--run-dir", default=None)
    report.add_argument("--output-dir", help="Derived report destination (default: reports).")
    smoke_command = commands.add_parser("smoke", help="Optional explicit image/caption check; no dataset or scores.")
    smoke_command.add_argument("--image", required=True)
    smoke_command.add_argument("--caption", required=True)
    comparison = commands.add_parser("compare", help="Write the selected dataset's comparison JSON and CSV; read-only access to saved runs.")
    comparison.add_argument("--output-dir", help="Derived report destination (default: reports).")
    args = parser.parse_args(argv)
    if args.command != "compare" and not (args.model or args.config):
        parser.error("Choose --model or --config explicitly.")
    if args.command == "compare" and (args.model or args.config):
        parser.error("compare reads all three configured experiments; use report for one model.")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.version_info < (3, 13):
            raise RuntimeError("Baseline requires Python 3.13+ for bounded-memory tar scanning.")
        if args.command == "compare":
            from .reports import compare_results
            compare_results(ROOT, local_path(ROOT, args.output_dir) if args.output_dir else None,
                            {key: getattr(args, key) for key in ("annotations", "archive", "images_dir", "prepared_dir") if getattr(args, key) is not None}, dataset=args.dataset)
            return 0
        config = load_model_config(ROOT, model_id=args.model, path=args.config, dataset=args.dataset)
        for key in ("annotations", "archive", "images_dir", "prepared_dir", "base_url", "server_log"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        if hasattr(args, "run_dir"):
            args.run_dir = args.run_dir or config["run_dir"]
        if args.command == "smoke":
            return smoke(config, args.image, args.caption)
        if args.command == "prepare":
            freeze_protocol(config)
            prepare(ROOT, config)
            checked_manifest(config)
        else:
            directory = local_path(ROOT, args.run_dir)
            if args.command == "run":
                return run_baseline(directory, config, args.resume, output_dir=local_path(ROOT, args.output_dir) if args.output_dir else None)
            elif args.command == "status":
                print(json.dumps(inspect_run(directory, config), indent=2))
            elif args.command == "report":
                from .reports import write_report
                report = write_report(directory, config, ROOT, local_path(ROOT, args.output_dir) if args.output_dir else None)
                print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
