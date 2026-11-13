"""Durable local writes, including Windows readers that temporarily deny rename."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def write_json(path, value):
    """Keep the previous valid JSON until atomic replacement succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # WinError 5/32/33 can be caused by a reader without delete-sharing.
        delays = (0.05, 0.1, 0.2, 0.4, 0.8)
        for attempt in range(len(delays) + 1):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Do not hide the original error if a scanner also holds the temp file.


def log_event(directory, event, **details):
    """An append-only event log; logging failures must not discard model responses."""
    value = {"at_utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    try:
        with (Path(directory) / "events.jsonl").open("ab") as handle:
            handle.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        print(f"WARNING: event log unavailable: {exc}; event={value}", file=sys.stderr)
