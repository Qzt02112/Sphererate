#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.evaluation.evaluate import evaluate_prediction_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    output_dir = args.output_dir or Path(config["paths"]["reports_root"]) / "evaluation"
    summary = evaluate_prediction_dir(processed_root, args.pred_dir, split=args.split, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
