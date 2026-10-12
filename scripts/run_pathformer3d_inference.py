#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.select_kernel_v2_pathformer3d import _generate, _load_model, _official_images
from scripts.train_kernel_v2_pathformer3d import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a selected Pathformer3D checkpoint on any matched ERP view.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--split", choices=("val", "gate"), default="gate")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-scanpaths", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    CONFIG["start_pitch_deg"] = float(args.start_pitch_deg)
    CONFIG["start_yaw_deg"] = float(args.start_yaw_deg)
    checkpoint = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    dataset_json = (ROOT / args.dataset_json).resolve() if not Path(args.dataset_json).is_absolute() else Path(args.dataset_json)
    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    device = torch.device(args.device)
    images = _official_images(dataset_json, args.split)
    if len(images) != int(args.expected_count):
        raise ValueError(f"Expected {args.expected_count} images, found {len(images)}")
    state = torch.load(checkpoint, map_location="cpu")
    model = _load_model(checkpoint, device)
    _generate(
        model,
        images,
        run_dir / "scanpaths",
        int(args.n_scanpaths),
        device,
        sampling_seed=int(args.seed),
    )
    metadata = {
        "method": "Pathformer3D",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_epoch": int(state["epoch"]),
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": hashlib.sha256(dataset_json.read_bytes()).hexdigest(),
        "split": str(args.split),
        "content_count": len(images),
        "scanpaths_per_content": int(args.n_scanpaths),
        "steps": int(CONFIG["max_length"]),
        "coordinate_order": "normalized_yx",
        "start_mode": "declared",
        "start_pitch_deg": float(args.start_pitch_deg),
        "start_yaw_deg": float(args.start_yaw_deg),
        "sampling_seed": int(args.seed),
        "target_domain_adaptation": False,
        "test_target_loads": 0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
