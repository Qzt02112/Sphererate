#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.visualize import make_gt_sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out-dir", default="reports/f0_visuals")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    paths = make_gt_sheets(processed_root, args.split, args.out_dir, limit=args.limit)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
