#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.evaluation.human import evaluate_human_upper_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--protocol", choices=["leave_one_observer", "split_half"], default="leave_one_observer")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    out_dir = args.output_dir or Path(config["paths"]["reports_root"]) / "f0_human_upper_bound" / args.protocol / args.split
    summary = evaluate_human_upper_bound(
        processed_root=processed_root,
        split=args.split,
        num_bins=int(config["data"]["num_bins"]),
        sigma_px=float(config["data"]["sigma_px"]),
        output_dir=out_dir,
        max_images=args.max_images,
        protocol=args.protocol,
        repeats=args.repeats,
        seed=int(config["project"].get("seed", 20260708)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
