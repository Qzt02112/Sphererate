#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_eh3_gate import _predict_learned
from sphererate.data.early_dataset import EarlyHorizonDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen SphereHorizon predictions for figures.")
    parser.add_argument("--split", choices=("val", "gate"), default="gate")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split=args.split,
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    for seed in (20260710, 20260711, 20260712):
        predictions, _ = _predict_learned(
            ROOT / f"runs/kernel_v2_candidate_correct_seed{seed}/checkpoints/best.pt",
            samples,
            torch.device(args.device),
        )
        output = ROOT / f"runs/kernel_v2_candidate_correct_seed{seed}/predictions/{args.split}"
        output.mkdir(parents=True, exist_ok=True)
        for image_id, maps in predictions.items():
            np.savez_compressed(output / f"{image_id}.npz", cumulative=maps.astype(np.float32))
        print(f"seed={seed} split={args.split} contents={len(predictions)} -> {output}")


if __name__ == "__main__":
    main()
