#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.train import _loader, load_model_for_inference, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    model, config, device = load_model_for_inference(args.checkpoint, args.device)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    loader = _loader(config, args.split, processed_root, augment=False)
    out_dir = args.out_dir or Path(args.checkpoint).parents[1] / "predictions" / args.split
    save_predictions(model, loader, device, out_dir)
    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
