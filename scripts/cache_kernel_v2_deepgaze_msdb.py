#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "deepgaze-msdb"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(THIRD_PARTY))

from deepgaze_pytorch import DeepGazeMSDB
from sphererate.baselines.deepgaze_msdb import normalize_spherical, spherical_area_weights
from sphererate.utils import ensure_dir


SOURCE_COMMIT = "90c49a554ce334add49bd54825679f61d0ac32ba"
REFERENCE_PPD = 30.0
ROLL_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
EXPECTED_COUNTS = {"train": 50, "val": 9, "gate": 13, "test": 13}


def _manifest_rows(root: Path, split: str, expected_count: int) -> list[dict]:
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = sorted((row for row in rows if str(row["split"]) == split), key=lambda row: str(row["image_id"]))
    if len(selected) != int(expected_count):
        raise ValueError(f"Expected {expected_count} {split} contents, got {len(selected)}")
    return selected


@torch.inference_mode()
def _predict(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> tuple[np.ndarray, float, float]:
    height, width = rgb.shape[:2]
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.uint8)
    centerbias = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    outputs = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for fraction in ROLL_FRACTIONS:
        shift = int(round(width * fraction)) % width
        current = torch.roll(image, shifts=shift, dims=-1)
        prediction = model(current, centerbias, pixel_per_dva=REFERENCE_PPD, dataset=None).exp()[0]
        outputs.append(torch.roll(prediction, shifts=-shift, dims=-1).float().cpu().numpy())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / (2**20)
    area = spherical_area_weights(height, width)
    return normalize_spherical(np.mean(outputs, axis=0), area), elapsed, peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen DeepGaze MSDB predictions for the kernel-v2 split.")
    parser.add_argument("--split", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--output-root", default="runs/kernel_v2_deepgaze_msdb/raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    data_root = ROOT / args.data_root
    rows = _manifest_rows(
        data_root,
        args.split,
        int(args.expected_count if args.expected_count is not None else EXPECTED_COUNTS[args.split]),
    )
    output = ensure_dir(ROOT / args.output_root / args.split)
    map_root = ensure_dir(output / "maps")
    device = torch.device(args.device)
    model = DeepGazeMSDB(pretrained=True).to(device)
    # The official feature branch's train()/eval() currently returns None, so do not chain this call.
    model.eval()

    timings, peaks = [], []
    for index, row in enumerate(rows, start=1):
        destination = map_root / f"{row['image_id']}.npy"
        if destination.exists() and not args.overwrite:
            cached = np.load(destination)
            if cached.shape != (256, 512) or not np.all(np.isfinite(cached)):
                raise ValueError(f"Invalid existing cache: {destination}")
            print(json.dumps({"index": index, "image_id": row["image_id"], "cached": True}), flush=True)
            continue
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (256, 512, 3):
            raise ValueError(f"Frozen MSDB adapter expects 256x512 RGB, got {rgb.shape}")
        prediction, elapsed, peak = _predict(model, rgb, device)
        np.save(destination, prediction.astype(np.float32))
        timings.append(elapsed)
        peaks.append(peak)
        print(
            json.dumps(
                {"index": index, "image_id": row["image_id"], "seconds": elapsed, "peak_mib": peak}
            ),
            flush=True,
        )

    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "deepgazemsdb.pth"
    metadata = {
        "method": "DeepGaze MSDB (ICCV 2025)",
        "split": args.split,
        "content_count": len(rows),
        "source": "https://github.com/matthias-k/deepgaze/tree/feature-deepgaze-msdb",
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "adapter": {
            "input": "native 256x512 ERP",
            "reference_pixel_per_dva": REFERENCE_PPD,
            "reference_ppd_is_physical_hmd_measurement": False,
            "yaw_roll_fractions": list(ROLL_FRACTIONS),
            "fusion": "arithmetic mean after undoing each circular yaw roll",
            "official_dataset_parameters": "mean across the five published datasets",
            "official_centerbias_input": "uniform",
            "metric_density_normalization": "spherical area",
        },
        "adapter_limit": (
            "MSDB is a planar-image model. The frozen reference ppd avoids pretending that the full ERP has a "
            "single physical HMD ppd; this row is an explicit ERP adaptation, not native spherical inference."
        ),
        "mean_seconds_new_items": float(np.mean(timings)) if timings else None,
        "max_peak_mib_new_items": float(np.max(peaks)) if peaks else None,
        "training_target_loads": 0,
        "test_target_loads": 0,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(output / "metadata.json")}, indent=2))


if __name__ == "__main__":
    main()
