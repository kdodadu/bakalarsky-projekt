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
import traceback

from .client import LocalClient, interpret, make_payload
from .data import image_candidates, local_path, prepare, select_test_records, sha256_file
from .evaluation import evaluate
from .storage import log_event, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "results/test_merged_balanced_baseline"
FROZEN_FILES = ("src/newsclip_pilot/client.py", "src/newsclip_pilot/evaluation.py")
GENERATION_KEYS = ("seed", "base_url", "model", "temperature", "repeat_penalty",
                   "chat_template_kwargs", "max_tokens", "timeout_seconds")
SNAPSHOT_EVERY = 25
# Audited predecessor: same frozen client/parser/metrics; only orchestration is repaired.
LEGACY_RUNNER_SHA256 = "2deaf6ec23fb5c9a5cbd20bcebe86ff39b88e92b87e8f7fe33d2e9bee366dd38"
PREVIOUS_RUNNER_SHA256 = "24ff9028d964eb3bf627c70e929d4ed11b39afefee6017c446e47c65b202c177"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


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
    return {"contract": contract, "fingerprint": fingerprint(contract)}


def checked_manifest(config, root=ROOT):
    manifest_path = local_path(root, config["prepared_dir"]) / "manifest.json"
    manifest = read_json(manifest_path)
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
        if not image.is_relative_to(local_path(root, config["prepared_dir"]) / "images"):
            raise ValueError("Image is outside prepared test images.")
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
    metadata = {"schema_version": 1, "kind": "newsclippings_full_test_baseline", "split": "test",
                "created_at_utc": now(), "status": "ready", "complete": False,
                "n_completed": 0, "expected_records": len(manifest["records"]),
                "config": config, "protocol": protocol, "prompt": protocol["contract"]["prompt"],
                "manifest_sha256": sha256_file(manifest_path), "input_manifest_sha256": fingerprint(manifest),
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "runner_sha256": sha256_file(Path(__file__)),
                "generation_order": "official annotation order, no sampling or shuffle",
                "sessions": [], "recoveries": [], "server": None}
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
            for key in ("id", "image_id", "pair_index", "true_label"):
                if result[key] != expected[index][key]:
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


def process_pending(directory, metadata, rows, records, client, stop, root=ROOT):
    indices = {row["annotation_index"] for row in rows}
    if len(indices) != len(rows) or len(rows) != metadata["expected_records"]:
        raise ValueError("Queue must contain every expected unique test annotation.")
    done = {r["annotation_index"] for r in records}
    if len(done) != len(records) or not done <= indices:
        raise ValueError("Saved records are duplicated or absent from the queue.")
    config = metadata["config"]
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
            payload = make_payload(config, local_path(root, row["image_path"]), text)
            started_at = now()
            metadata["phase"] = "request"
            log_event(directory, "request_started", annotation_index=row["annotation_index"], saved=len(records))
            response = client.request("/v1/chat/completions", payload)
            result = {"annotation_index": row["annotation_index"], "pair_index": row["pair_index"],
                      "id": row["id"], "image_id": row["image_id"], "true_label": row["true_label"],
                      "started_at_utc": started_at, **response, **interpret(response)}
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
                metadata["status"] = "stopped_on_request_error"
                metadata["stop_reason"] = {"kind": "request_error", "annotation_index": row["annotation_index"],
                                           "http_status": result["http_status"], "error": result["error"]}
                print("Request error saved and counted. Stopping to avoid repeated infrastructure failures; resume skips this saved record.", flush=True)
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


def verify_run(directory, config, protocol, manifest_path, manifest):
    metadata = read_json(directory / "run.json")
    if metadata.get("kind") != "newsclippings_full_test_baseline":
        raise ValueError("This is not a baseline run; pilot directories are read-only.")
    # Historical runs include a pilot path and hashes. Ignore only that obsolete
    # dependency; compare experiment settings and the saved contract itself.
    settings = lambda c: {k: v for k, v in c.items() if k not in ("reference_pilot", "protocol")}
    saved_protocol = metadata["protocol"]
    if (settings(metadata["config"]) != settings(config)
            or saved_protocol["contract"] != protocol["contract"]
            or saved_protocol.get("fingerprint") != protocol.get("fingerprint")
            or metadata["prompt"] != protocol["contract"]["prompt"]):
        raise ValueError("Settings differ from this run's saved snapshot. Resume refused.")
    if metadata.get("server") and config.get("protocol", {}).get("server") != server_signature(metadata["server"]["props"]["response"]):
        raise ValueError("Configured server protocol differs from the saved run.")
    current_runner = sha256_file(Path(__file__))
    previous_runner = metadata.get("active_runner_sha256", metadata["runner_sha256"])
    if previous_runner not in (current_runner, LEGACY_RUNNER_SHA256, PREVIOUS_RUNNER_SHA256):
        raise ValueError("Unknown runner revision; automatic compatibility has not been verified.")
    saved_manifest = read_json(directory / "input_manifest.json")
    if metadata["manifest_sha256"] != sha256_file(manifest_path) or metadata["input_manifest_sha256"] != fingerprint(saved_manifest) or saved_manifest != manifest:
        raise ValueError("Run or prepared input manifest changed; resume refused.")
    if metadata["expected_records"] != len(manifest["records"]):
        raise ValueError("Expected record count differs from the verified manifest.")
    return metadata, previous_runner, current_runner


def inspect_run(directory, config, root=ROOT):
    """Read-only end-to-end validation, with counts computed from the journal."""
    protocol = freeze_protocol(config, root)
    manifest_path, manifest = checked_manifest(config, root)
    metadata, old_runner, current_runner = verify_run(directory, config, protocol, manifest_path, manifest)
    records, _ = load_journal(directory / "predictions.jsonl", manifest["records"], repair=False)
    done = {row["annotation_index"] for row in records}
    pending = [row["annotation_index"] for row in manifest["records"] if row["annotation_index"] not in done]
    selection = read_json(local_path(root, config["prepared_dir"]) / "selection.json")
    sessions = metadata.get("sessions", [])
    return {"directory": str(directory), "stored_status": metadata["status"],
            "counts": {"raw_test": manifest["audit"]["records"], "prepared_selection": len(selection["records"]),
                       "prepared_manifest": len(manifest["records"]), "run_queue": len(manifest["records"]),
                       "saved_unique_records": len(done), "remaining": len(pending),
                       "previously_saved_at_last_session_start": sessions[-1]["already_saved"] if sessions else 0},
            "next_annotation_index": pending[0] if pending else None,
            "stop_reason": metadata.get("stop_reason") or (sessions[-1].get("error") if sessions else None),
            "compatible_runner_upgrade": old_runner != current_runner,
            "frozen_protocol_verified": True, "images_verified": True,
            "metrics": evaluate(records, metadata["expected_records"])}


def run_baseline(directory, config, resume=False, root=ROOT):
    if (directory / "run.json").exists() and read_json(directory / "run.json").get("kind") != "newsclippings_full_test_baseline":
        raise ValueError("This directory is not a baseline run; it will not be modified.")
    protocol = freeze_protocol(config, root)
    manifest_path, manifest = checked_manifest(config, root)
    if resume and (directory / "run.json").exists():
        previous, _, _ = verify_run(directory, config, protocol, manifest_path, manifest)
        existing, _ = load_journal(directory / "predictions.jsonl", manifest["records"], repair=False)
        if len(existing) == previous["expected_records"]:
            print(json.dumps(evaluate(existing, previous["expected_records"]), indent=2))
            print("Already complete; no server requests or result writes made.")
            return 0
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
                        "reason": "Compatible orchestration update; frozen inference/metrics unchanged"}
            metadata.setdefault("runner_revisions", []).append(revision)
            metadata["active_runner_sha256"] = current_runner
            log_event(directory, "compatible_runner_upgrade", **revision)
        metrics = save_progress(directory, metadata, records)
        if metrics["complete"]:
            metadata["status"] = "completed"
            save_progress(directory, metadata, records)
            print(f"Already complete: {len(records)}/{metadata['expected_records']}; no server requests made.")
            return 0
        session = {"started_at_utc": now(), "resumed": resume, "already_saved": len(records),
                   "queued": metadata["expected_records"] - len(records), "runner_sha256": current_runner}
        metadata["sessions"].append(session)
        metadata["status"] = "starting"
        metadata["stop_reason"] = None
        metadata["phase"] = "server_preflight"
        log_event(directory, "session_started", **session)
        stop = StopRequest()
        previous_handler = signal.signal(signal.SIGINT, stop.handle_signal)
        try:
            client, metadata["server"] = check_server(config, root, metadata.get("server"))
            print(f"Pending: {len(manifest['records']) - len(records)} / {metadata['expected_records']}. Directory: {directory}", flush=True)
            process_pending(directory, metadata, manifest["records"], records, client, stop, root)
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
        return 0 if metrics["complete"] else 130 if metadata["status"] == "interrupted" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen full NewsCLIPpings test baseline; no training or VERITE.")
    parser.add_argument("--config", default="configs/test.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Prepare all 7264 test records without inference.")
    run = commands.add_parser("run", help="Start or resume the complete test baseline.")
    run.add_argument("--run-dir", default=DEFAULT_RUN)
    run.add_argument("--resume", action="store_true")
    status = commands.add_parser("status", help="Verify data, queue and journal; report progress without inference or writes.")
    status.add_argument("--run-dir", default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        if sys.version_info < (3, 13):
            raise RuntimeError("Baseline requires Python 3.13+ for bounded-memory tar scanning.")
        config = read_json(local_path(ROOT, args.config))
        if args.command == "prepare":
            freeze_protocol(config)
            prepare(ROOT, config)
            checked_manifest(config)
        else:
            directory = local_path(ROOT, args.run_dir)
            if args.command == "run":
                return run_baseline(directory, config, args.resume)
            elif args.command == "status":
                print(json.dumps(inspect_run(directory, config), indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
