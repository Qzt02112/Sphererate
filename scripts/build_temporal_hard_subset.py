#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.evaluation.hard_subset import select_hard_rows, temporal_hardness_scores, write_hard_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True, help="Path to write selected image ids.")
    args = parser.parse_args()

    if args.top_k is not None and args.top_fraction is not None:
        raise ValueError("Use either --top-k or --top-fraction, not both.")
    if args.top_fraction is not None and not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top-fraction must be in (0, 1].")

    cfg = load_config(args.config)
    processed_root = Path(cfg["paths"]["processed_root"]) / cfg["data"]["processed_name"]
    rows = temporal_hardness_scores(processed_root, split=args.split)
    selected = select_hard_rows(rows, top_k=args.top_k, top_fraction=args.top_fraction)
    write_hard_subset(rows, selected, csv_path=args.csv, id_path=args.out)

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.out}")
    print(f"Selected {len(selected)}/{len(rows)} {args.split} images")
    if selected:
        print("Top ids: " + ", ".join(str(row["image_id"]) for row in selected[:10]))


if __name__ == "__main__":
    main()
