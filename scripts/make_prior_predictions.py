#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.baselines.priors import make_prior_predictions
from sphererate.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--kind", choices=["uniform", "equator", "center", "train_average"], required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    out_dir = args.out_dir or Path(config["paths"]["runs_root"]) / f"prior_{args.kind}" / "predictions" / args.split
    out = make_prior_predictions(processed_root, out_dir, args.kind, args.split)
    print(f"Saved {args.kind} prior predictions to {out}")


if __name__ == "__main__":
    main()
