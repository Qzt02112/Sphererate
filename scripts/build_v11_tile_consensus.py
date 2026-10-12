#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_horizon import raw_dwell_topk_consensus
from sphererate.data.early_view import read_manifest
from sphererate.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val observer-consensus tile labels.")
    parser.add_argument(
        "--processed-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/v11_raw_tile_consensus_seed20260805",
    )
    parser.add_argument("--observer-splits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    processed_root = Path(args.processed_root)
    output_root = ensure_dir(args.output_root)
    rows = [
        row
        for row in read_manifest(processed_root / "manifest.jsonl")
        if str(row["split"]) in {"train", "val"}
    ]
    horizons = (0.5, 1.0, 3.0, 5.0)
    budgets = (5.0, 10.0, 20.0)
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    for row in rows:
        with np.load(row["fixations"]) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        consensus = raw_dwell_topk_consensus(
            fixations,
            horizons_s=horizons,
            budgets_percent=budgets,
            tile_rows=8,
            tile_cols=16,
            observer_splits=int(args.observer_splits),
            seed=int(args.seed),
            key=str(row["canonical_content_id"]),
        )
        np.savez_compressed(
            output_root / f"{row['image_id']}.npz",
            topk_consensus=consensus,
            horizons_s=np.asarray(horizons, dtype=np.float32),
            budgets_percent=np.asarray(budgets, dtype=np.float32),
        )
        split_counts[str(row["split"])] += 1
    metadata = {
        "source": str(processed_root),
        "splits": split_counts,
        "forbidden_splits": ["gate", "test", "sitzmann"],
        "horizons_s": list(horizons),
        "budgets_percent": list(budgets),
        "tile_grid": [8, 16],
        "observer_splits": int(args.observer_splits),
        "half_samples_per_content": int(args.observer_splits) * 2,
        "seed": int(args.seed),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
