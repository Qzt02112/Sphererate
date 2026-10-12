#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.future_view import build_future_saliency_data_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a zero-leakage future-saliency data view.")
    parser.add_argument("--etdk2-root", type=Path, required=True)
    parser.add_argument("--salient360-root", type=Path, required=True)
    parser.add_argument("--sitzmann-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    out = build_future_saliency_data_view(
        etdk2_root=args.etdk2_root,
        salient360_root=args.salient360_root,
        sitzmann_root=args.sitzmann_root,
        out_root=args.out_root,
        split_file=args.split_file,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(f"Prepared future-saliency data view at {out}")


if __name__ == "__main__":
    main()
