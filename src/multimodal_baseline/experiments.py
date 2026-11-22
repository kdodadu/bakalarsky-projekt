"""Native model identity checks; standard library only."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
from urllib.parse import urlsplit

from .data import local_path, sha256_file


def experiment_identity(experiment):
    """Keep all model/settings evidence, excluding descriptions and local launch locations."""
    result = json.loads(json.dumps(experiment))
    for key in ("capacity_validation_note", "revision_provenance", "historical_identity_note", "response_format_note"):
        result.pop(key, None)
    command = result.get("launch_command")
    if command:
        # Exact asset bytes are pinned separately. Moving an executable, weights,
        # logs or endpoint does not change inference. Every other flag stays strict.
        paths = {"--log-file", "-m", "--model", "--mmproj", "-hf", "--hf-repo", "--hf-file", "--host", "--port", "--alias", "-a"}
        normalized, index = [], 1
        while index < len(command):
            token = command[index]
            if token in paths:
                index += 2
            elif token == "--offline":
                index += 1
            else:
                normalized.append(token)
                index += 1
        result["launch_command"] = normalized
    return result


def gguf_metadata(path):
    """Read selected GGUF metadata, skipping vocabulary arrays and tensor data."""
    widths = {0: (1, "B"), 1: (1, "b"), 2: (2, "H"), 3: (2, "h"),
              4: (4, "I"), 5: (4, "i"), 6: (4, "f"), 7: (1, "?"),
              10: (8, "Q"), 11: (8, "q"), 12: (8, "d")}
    with Path(path).open("rb") as handle:
        def number(fmt):
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))[0]

        def value(kind, keep):
            if kind in widths:
                size, fmt = widths[kind]
                if keep:
                    return number(fmt)
                handle.seek(size, 1)
            elif kind == 8:
                size = number("Q")
                if keep:
                    if size > 4 * 1024 * 1024:
                        raise ValueError("Oversized GGUF metadata string")
                    return handle.read(size).decode("utf-8")
                handle.seek(size, 1)
            elif kind == 9:
                item_kind, count = number("I"), number("Q")
                if item_kind in widths:
                    handle.seek(widths[item_kind][0] * count, 1)
                else:
                    for _ in range(count):
                        value(item_kind, False)
            else:
                raise ValueError(f"Unknown GGUF metadata type: {kind}")

        if handle.read(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError(f"Not a supported GGUF: {path}")
        number("Q")  # tensor count
        count = number("Q")
        result = {}
        for _ in range(count):
            key = value(8, True)
            kind = number("I")
            keep = (key in ("general.architecture", "general.name", "general.file_type", "tokenizer.chat_template")
                    or key.startswith("clip.")) and kind != 9
            data = value(kind, keep)
            if keep:
                result[key] = data
        return result


def observe_process(base_url):
    """Read the actual listener's Windows command line. Never launch/stop a server."""
    port = urlsplit(base_url).port or 80
    command = f"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$owner = Get-NetTCPConnection -State Listen -LocalPort {int(port)} | Select-Object -First 1 -ExpandProperty OwningProcess
$process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $owner)
[pscustomobject]@{{pid=$owner;command_line=$process.CommandLine;created_at=$process.CreationDate.ToString('o')}} | ConvertTo-Json -Compress
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, timeout=30, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Cannot inspect endpoint process. Run the evaluation from your normal Windows terminal with access to Get-NetTCPConnection/Get-CimInstance. " + result.stderr[-500:])
    process = json.loads(result.stdout.lstrip("\ufeff"))
    if not process.get("command_line"):
        raise RuntimeError("Endpoint command line unavailable; model identity cannot be verified.")
    return process


def command_options(command_line):
    tokens = [s.strip('"') for s in shlex.split(command_line, posix=False)]
    result = {}
    for i, token in enumerate(tokens):
        if token.startswith("-"):
            if "=" in token:
                key, value = token.split("=", 1)
                result[key] = value.strip('"')
            else:
                result[token] = tokens[i + 1] if i + 1 < len(tokens) else None
    return result


def checked_asset(path, expected):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Loaded weight file is unavailable locally: {path}")
    digest = sha256_file(path)
    # llama.cpp caches may prefix filenames with the repository name. Identity is
    # established by bytes, not by assuming the cache uses the HF basename.
    record = next((item for item in expected if item["sha256"] == digest), None)
    if record is None:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}; expected pinned repository weights.")
    return {"path": str(path), "filename": path.name, "repository_filename": record["filename"],
            "sha256": digest, "bytes": path.stat().st_size,
            "metadata": gguf_metadata(path)}


def verify_native_template(embedded, served, pinned_sha256=None):
    """Accept exact text or the observed server removal of one terminal LF only."""
    normalization = "exact" if embedded else "native fallback"
    if embedded and embedded != served:
        if not embedded.endswith("\n") or embedded[:-1] != served:
            raise RuntimeError("Server template differs from the native template embedded in the loaded GGUF.")
        normalization = "one terminal LF removed by server"
    embedded_hash = hashlib.sha256(embedded.encode()).hexdigest() if embedded else None
    if pinned_sha256 and embedded_hash != pinned_sha256:
        raise RuntimeError("Embedded native template differs from the pinned local GGUF template.")
    return {"embedded_sha256": embedded_hash, "served_sha256": hashlib.sha256(served.encode()).hexdigest(),
            "normalization": normalization}


def verify_model_server(config, server, root, saved_server=None):
    experiment = config["experiment"]
    props = server["props"]["response"]
    if not isinstance(props, dict) or not props.get("modalities", {}).get("vision"):
        raise RuntimeError("Server does not report image input capability.")
    settings = props.get("default_generation_settings", {})
    if settings.get("n_ctx") != experiment["context_size"] or props.get("total_slots") != 1:
        raise RuntimeError(f"Expected ctx-size {experiment['context_size']} and parallel 1; check the server launch command.")
    # New experiments can pin the entire observed reference settings rather
    # than silently inheriting architecture/backend-specific server defaults.
    for key, expected in experiment.get("server_requirements", {}).items():
        if props.get(key) != expected:
            raise RuntimeError(f"Server {key} differs from the configured reference settings; no model requests sent.")
    process = observe_process(config["base_url"])
    options = command_options(process["command_line"])
    image_processing = experiment.get("image_processing")
    if image_processing:
        for flag,key in (("--image-min-tokens","image_min_tokens"),("--image-max-tokens","image_max_tokens")):
            if options.get(flag)!=str(image_processing[key]):
                raise RuntimeError(f"Expected explicit {flag} {image_processing[key]}; image processing differs from the experiment.")
    expected_options = command_options(" ".join(experiment.get("launch_command", [])))
    for flag in ("--device", "--gpu-layers", "--fit"):
        if flag in expected_options and options.get(flag) != expected_options[flag]:
            raise RuntimeError(f"Expected explicit {flag} {expected_options[flag]}; server placement/backend differs.")
    if any(option in options for option in ("--chat-template", "--chat-template-file", "--chat-template-kwargs")):
        raise RuntimeError("Custom chat template overrides are not allowed; use the native model template.")
    response_format = experiment.get("response_reasoning_format")
    if response_format and (options.get("--reasoning-format") != response_format or "--skip-chat-parsing" in options):
        raise RuntimeError(f"Restart this model's server with --reasoning-format {response_format} and native chat parsing. The current API response format is incompatible with exact MATCH/MISMATCH parsing.")
    if experiment["reasoning_budget"] == 0 and options.get("--reasoning-budget") != "0":
        raise RuntimeError("Qwen requires the observed --reasoning-budget 0 flag.")
    model = checked_asset(props.get("model_path", ""), [experiment["gguf"]])
    template = props.get("chat_template")
    if not isinstance(template, str) or not template:
        raise RuntimeError("Server chat template is unavailable.")
    embedded = model["metadata"].get("tokenizer.chat_template")
    template_verification = verify_native_template(embedded, template, experiment.get("native_template_sha256"))
    log_path = local_path(root, config["server_log"])
    actual_log = options.get("--log-file")
    if not actual_log or local_path(root, actual_log) != log_path:
        raise RuntimeError(f"Endpoint must use --log-file {config['server_log']} so the loaded mmproj can be verified.")
    if process.get("created_at") and log_path.stat().st_mtime < datetime.fromisoformat(process["created_at"]).timestamp():
        raise RuntimeError("Startup log predates the endpoint process; cannot verify its loaded mmproj.")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"loaded multimodal model,\s*['\"]([^'\"\r\n]+)['\"]", log)
    if not matches:
        raise RuntimeError("Startup log has no 'loaded multimodal model' path; cannot verify the actual mmproj.")
    projector = checked_asset(local_path(root, matches[-1]), experiment["mmproj"])
    if not props.get("build_info"):
        raise RuntimeError("Server build/version metadata unavailable.")
    # Hashes and full /props snapshot freeze both architecture-specific processing
    # and server defaults on resume. PID, process start time and local paths may change.
    identity = {"original_model_id": experiment["original_model_id"], "gguf_repository": experiment["gguf_repository"],
                "repository_revision": experiment["revision"], "gguf_sha256": model["sha256"],
                "mmproj_sha256": projector["sha256"], "build_info": props["build_info"],
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "default_generation_settings": settings, "context_size": settings["n_ctx"],
                "parallel": props["total_slots"], "reasoning_budget": experiment["reasoning_budget"]}
    if response_format:
        identity["response_reasoning_format"] = response_format
    if image_processing:
        identity["image_processing"] = image_processing
    if saved_server and saved_server.get("identity") != identity:
        raise RuntimeError("Loaded model, mmproj, template, context or server defaults changed since this run. Resume refused.")
    # Avoid storing authentication arguments if a user launches with an explicit token.
    process["command_line"] = re.sub(r"(--hf-token|-hft)\s+(\"[^\"]*\"|\S+)", r"\1 [REDACTED]", process["command_line"])
    server.update(identity=identity, loaded_files={"gguf": model, "mmproj": projector}, observed_process=process,
                  template_verification=template_verification,
                  expected_launch_command=experiment["launch_command"],
                  native_image_preprocessing="llama.cpp mtmd for this GGUF/mmproj; differs by architecture; see file metadata and /props",
                  template_source="embedded GGUF" if embedded else "native llama.cpp fallback for the loaded architecture")
    return server
