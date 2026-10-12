#!/usr/bin/env python
from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.download_etdk2 import download_etdk2


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the official ET-DK2 v3 Zenodo release.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(os.path.expandvars("${SCANDMM_ROOT}/data/raw/etdk2")),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    record = download_etdk2(args.out_root, timeout=args.timeout, extract=not args.no_extract)
    print(json.dumps({
        "out_root": str(args.out_root),
        "record_id": record["record_id"],
        "version": record["version"],
        "participant_files": len(record["participant_files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
