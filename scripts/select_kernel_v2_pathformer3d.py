#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/Pathformer3D"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

import datasets.dataloder as official_data
from metircs.suppor_lib import xyz2sphere
from modules.mdn3d import mixture_probability

from scripts.evaluate_eh6_scandmm import _macro, _score
from scripts.train_kernel_v2_pathformer3d import CONFIG, _model, _start_xyz
from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset


def _lat_lon_to_normalized_yx(sphere: torch.Tensor) -> np.ndarray:
    values = sphere.detach().cpu().numpy()
    y = (values[..., 0] + 90.0) / 180.0
    x = (values[..., 1] + 180.0) / 360.0
    return np.stack((y, x), axis=-1).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _sample_scanpaths(
    model: torch.nn.Module,
    image: torch.Tensor,
    n_paths: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    enc_inputs = model.feature_extrator(image[None].to(device))
    enc_outputs, _ = model.encoder(enc_inputs, None)
    enc_outputs = enc_outputs.repeat(int(n_paths), 1, 1)
    dec_inputs = _start_xyz(device).view(1, 1, 3).repeat(int(n_paths), 1, 1)
    sphere_grid = model.mdn.xyz_t.to(device)
    grid_points = sphere_grid[0].transpose(0, 1)
    for _ in range(int(CONFIG["max_length"])):
        length = int(dec_inputs.shape[1])
        dec_outputs, _, _ = model.decoder(
            enc_outputs,
            dec_inputs,
            enc_masks=None,
            dec_masks=torch.zeros((int(n_paths), length), device=device),
        )
        pis, mus, sigmas, rhos = model.mdn(dec_outputs[:, -1:])
        probabilities = mixture_probability(
            pis, mus, sigmas, rhos, sphere_grid.unsqueeze(1)
        )[:, 0]
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        totals = probabilities.sum(dim=1, keepdim=True)
        if torch.any(totals <= 0):
            raise FloatingPointError("Pathformer3D produced an invalid sampling distribution")
        indices = torch.multinomial(probabilities / totals, num_samples=1, replacement=True)[:, 0]
        next_point = grid_points[indices]
        dec_inputs = torch.cat((dec_inputs, next_point[:, None]), dim=1)
    sphere = xyz2sphere(dec_inputs[:, 1:].reshape(-1, 3).cpu()).reshape(
        int(n_paths), int(CONFIG["max_length"]), 2
    )
    # The common scanpath renderer consumes normalized ERP (y, x), not pixels.
    return _lat_lon_to_normalized_yx(sphere)


def _official_images(dataset_json: Path, split: str) -> dict[str, torch.Tensor]:
    official_data.paths["matched_data_dict_path"] = str(dataset_json)
    dataset = official_data.ScanPath360Dataset(
        phase=split,
        dataset_name="matched",
        image_input_resize=CONFIG["image_input_resize"],
        patch_size=CONFIG["patch_size"],
        max_length=CONFIG["max_length"],
    )
    return {str(dataset[index]["file_names"]): dataset[index]["imgs"] for index in range(len(dataset))}


def _generate(
    model: torch.nn.Module,
    images: dict[str, torch.Tensor],
    output: Path,
    n_paths: int,
    device: torch.device,
    sampling_seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    expected = set(images)
    existing = {path.stem for path in output.glob("*.npy")}
    def valid_cache(path: Path) -> bool:
        if not path.exists():
            return False
        values = np.load(path)
        return bool(
            values.shape == (int(n_paths), int(CONFIG["max_length"]), 2)
            and np.isfinite(values).all()
            and np.all(values >= 0.0)
            and np.all(values <= 1.0)
        )

    if existing == expected and all(valid_cache(output / f"{image_id}.npy") for image_id in expected):
        return
    if existing - expected:
        raise ValueError(f"Unexpected cached IDs: {sorted(existing-expected)}")
    torch.manual_seed(int(sampling_seed))
    torch.cuda.manual_seed_all(int(sampling_seed))
    for image_id in sorted(images):
        path = output / f"{image_id}.npy"
        if not valid_cache(path):
            np.save(path, _sample_scanpaths(model, images[image_id], n_paths, device))
        print(f"[Pathformer3D] {image_id}", flush=True)


def _predictions(scanpath_dir: Path, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    predictions = {}
    for sample in samples:
        image_id = str(sample["image_id"])
        predictions[image_id] = second_sampled_scanpaths_to_cumulative_maps(
            np.load(scanpath_dir / f"{image_id}.npy"),
            horizons_s=sample["horizons_s"].numpy().tolist(),
            height=int(sample["target"].shape[-2]),
            width=int(sample["target"].shape[-1]),
            renderer="sphere_chord",
            sigma_deg=2.0,
            coordinate_order="yx",
        )
    return predictions


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location=device)
    model = _model().to(device)
    model.load_state_dict(state["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-select and infer matched Pathformer3D checkpoints.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--selection-paths", type=int, default=50)
    parser.add_argument("--final-paths", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset-json",
        default="data/processed/kernel_v2_matched_scanpaths/pathformer3d/dataset.json",
    )
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--expected-val", type=int, default=9)
    parser.add_argument("--expected-gate", type=int, default=13)
    parser.add_argument("--expected-test", type=int, default=13)
    parser.add_argument("--required-final-epoch", type=int, default=40)
    parser.add_argument("--run-prefix", default="kernel_v2_pathformer3d_retrained")
    args = parser.parse_args()
    if args.seed not in {20260710, 20260711, 20260712}:
        raise ValueError("Seed is outside the frozen matched-baseline protocol")
    train_run = ROOT / args.train_run
    checkpoints = sorted((train_run / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints or int(torch.load(checkpoints[-1], map_location="cpu")["epoch"]) != int(
        args.required_final_epoch
    ):
        raise RuntimeError(
            f"Pathformer3D selection is forbidden before final epoch {args.required_final_epoch} exists"
        )
    CONFIG["start_pitch_deg"] = float(args.start_pitch_deg)
    CONFIG["start_yaw_deg"] = float(args.start_yaw_deg)
    dataset_json = (ROOT / args.dataset_json).resolve()
    os.chdir(ROOT)
    device = torch.device(args.device)
    val_dataset = EarlyHorizonDataset(args.data_root, split="val", target_mode="correct")
    val_samples = [val_dataset[index] for index in range(len(val_dataset))]
    if len(val_samples) != int(args.expected_val):
        raise ValueError(f"Frozen val split must contain {args.expected_val} rows")
    val_images = _official_images(dataset_json, "val")
    if set(val_images) != {str(sample["image_id"]) for sample in val_samples}:
        raise ValueError("Pathformer3D val images do not match the frozen saliency split")

    selection_root = train_run / "selection"
    records = []
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu")
        epoch = int(state["epoch"])
        scanpath_dir = selection_root / f"epoch_{epoch:04d}" / "scanpaths"
        model = _load_model(checkpoint, device)
        _generate(
            model,
            val_images,
            scanpath_dir,
            args.selection_paths,
            device,
            sampling_seed=args.seed + 913,
        )
        scores, _ = _score(_predictions(scanpath_dir, val_samples), val_samples, (5.0, 10.0, 20.0))
        metric = _macro(scores, ("kld_sph", "sim_sph", "cc_sph"))
        records.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                **metric,
            }
        )
        del model
        torch.cuda.empty_cache()
    selected = min(records, key=lambda row: (float(row["kld_sph"]), int(row["epoch"])))
    selection = {
        "seed": args.seed,
        "criterion": "val early-macro KLD_sph over 0.5/1/3 s",
        "selection_paths_per_content": args.selection_paths,
        "records": records,
        "selected": selected,
        "test_target_loads": 0,
    }
    (train_run / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    model = _load_model(Path(str(selected["checkpoint"])), device)
    final_splits = ("val", "test") if "standard_v3" in str(args.data_root) else ("val", "gate")
    for offset, split in enumerate(final_splits):
        images = _official_images(dataset_json, split)
        expected_count = {
            "val": int(args.expected_val),
            "gate": int(args.expected_gate),
            "test": int(args.expected_test),
        }[split]
        if len(images) != expected_count:
            raise ValueError(f"Frozen {split} split mismatch")
        final_run = ROOT / f"runs/{args.run_prefix}_seed{args.seed}_{split}"
        scanpath_dir = final_run / "scanpaths"
        _generate(
            model,
            images,
            scanpath_dir,
            args.final_paths,
            device,
            sampling_seed=args.seed + 1901 + offset,
        )
        metadata = {
            "method": "Pathformer3D",
            "seed": args.seed,
            "split": split,
            "selected_epoch": int(selected["epoch"]),
            "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "scanpaths_per_content": args.final_paths,
            "steps": int(CONFIG["max_length"]),
            "coordinate_order": "normalized_yx",
            "start_mode": "declared",
            "start_pitch_deg": float(CONFIG["start_pitch_deg"]),
            "start_yaw_deg": float(CONFIG["start_yaw_deg"]),
            "test_evaluated": bool(split == "test"),
            "test_used_for_selection": False,
        }
        (final_run / "inference_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"selected": selected, "selection": str(train_run / "selection.json")}, indent=2))


if __name__ == "__main__":
    main()
