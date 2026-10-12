#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config, set_by_dotted
from sphererate.data.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--processed-name", default=None)
    parser.add_argument("--split-file", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_images is not None:
        config = set_by_dotted(config, "data.max_images", args.max_images)
    if args.processed_name:
        config = set_by_dotted(config, "data.processed_name", args.processed_name)
    if args.split_file:
        config = set_by_dotted(config, "data.split_file", args.split_file)
    out = prepare_dataset(config, dataset=args.dataset)
    print(f"Prepared dataset at {out}")


if __name__ == "__main__":
    main()
