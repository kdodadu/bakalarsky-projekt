"""Shared, explicitly selected model baseline on the full NewsCLIPpings test."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsclip_pilot.baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
