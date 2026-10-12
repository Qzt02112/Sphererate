#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.baselines.postprocess import postprocess_scanpath_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanpath-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=6)
    parser.add_argument("--sigma-px", type=float, default=5.0)
    parser.add_argument("--order", choices=["yx", "xy"], default="yx")
    args = parser.parse_args()
    out = postprocess_scanpath_dir(
        args.scanpath_dir,
        args.out_dir,
        height=args.height,
        width=args.width,
        num_bins=args.num_bins,
        sigma_px=args.sigma_px,
        order=args.order,
    )
    print(f"Saved saliency predictions to {out}")


if __name__ == "__main__":
    main()
