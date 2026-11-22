"""Shared, explicitly selected model baseline on NewsCLIPpings or VERITE."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_baseline.baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
