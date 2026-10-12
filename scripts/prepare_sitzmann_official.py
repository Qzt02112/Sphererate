#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.sitzmann_official import build_sitzmann_early_horizon_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official start-aware Sitzmann early-horizon data.")
    parser.add_argument("--archive", default="data/sitzmann_raw_data.zip")
    parser.add_argument("--legacy-python", default="${LEGACY_PYTHON}")
    parser.add_argument("--interim-root", default="data/interim/sitzmann_official_vr_v1")
    parser.add_argument("--out-root", default="data/processed/sitzmann_official_256x512_spherical2deg_v1")
    parser.add_argument("--split-file", default="data/splits/sitzmann_official_contentdisjoint_seed20260713.json")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    if not args.skip_extract:
        command = [
            args.legacy_python,
            str(ROOT / "scripts/extract_sitzmann_official_legacy.py"),
            "--archive",
            args.archive,
            "--out-root",
            args.interim_root,
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    output = build_sitzmann_early_horizon_view(
        interim_root=args.interim_root,
        out_root=args.out_root,
        split_file=args.split_file,
    )
    info = json.loads((output / "dataset_info.json").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), **info}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
