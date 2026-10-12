#!/usr/bin/env python
"""Select HAT by validation KLD and generate matched population scanpaths."""
from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/HAT"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SOURCE))

from common.utils import transform_fixations
from scripts.evaluate_eh6_scandmm import _macro, _score
from scripts.train_kernel_v2_hat import build_model, hat_config
from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset


STANDARD_SEEDS = {20260710, 20260711, 20260712}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_transform(pa: Any) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((pa.im_h, pa.im_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _images(dataset: dict[str, Any], split: str, pa: Any) -> dict[str, torch.Tensor]:
    root = Path(str(dataset["_path"])).parent
    transform = _image_transform(pa)
    result = {}
    for record in dataset["images"][split]:
        with Image.open(root / "images" / str(record["file"])) as image:
            result[str(record["image_id"])] = transform(image.convert("RGB"))
    return result


def _repeat_encoded(encoded: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    dorsal, pos, mask, high = encoded
    dorsal_pos, scale = pos
    return (
        dorsal.repeat(1, count, 1),
        (dorsal_pos.repeat(1, count, 1), scale.repeat(1, count, 1)),
        mask,
        (high[0].repeat(count, 1, 1, 1), high[1].repeat(count, 1, 1, 1)),
    )


def _wrapped_ior_mask(x: torch.Tensor, y: torch.Tensor, height: int, width: int, radius: int) -> torch.Tensor:
    rows = torch.arange(height, device=x.device, dtype=torch.float32)[None, :, None]
    cols = torch.arange(width, device=x.device, dtype=torch.float32)[None, None, :]
    px = x[:, None, None] * float(width)
    py = y[:, None, None] * float(height)
    dx = torch.abs(cols - px)
    dx = torch.minimum(dx, float(width) - dx)
    dy = rows - py
    return ((dx * dx + dy * dy) <= float(radius * radius)).reshape(len(x), -1)


@torch.no_grad()
def _sample_chunk(
    model: torch.nn.Module,
    encoded: tuple[Any, ...],
    count: int,
    pa: Any,
    device: torch.device,
    start_x: float,
    start_y: float,
    steps: int,
    ior_radius: int,
) -> np.ndarray:
    dorsal, pos, mask, high = _repeat_encoded(encoded, count)
    normalized = torch.tensor([start_x, start_y], dtype=torch.float32).view(1, 1, 2).repeat(count, 1, 1)
    normalized = normalized.to(device)
    visited = (
        _wrapped_ior_mask(
            normalized[:, 0, 0], normalized[:, 0, 1], pa.im_h, pa.im_w, radius=ior_radius
        )
        if ior_radius > 0
        else torch.zeros(count, pa.im_h * pa.im_w, dtype=torch.bool, device=device)
    )
    task_ids = torch.zeros(count, dtype=torch.long, device=device)
    outputs = []
    for _ in range(int(steps)):
        padding = torch.zeros(normalized.shape[:2], dtype=torch.float32)
        low, high_seq = transform_fixations(
            normalized.cpu(), padding, pa, False, return_highres=True
        )
        prediction = model.decode_and_predict(
            dorsal.clone(),
            pos,
            mask,
            high,
            low.to(device),
            torch.zeros_like(low, dtype=torch.bool, device=device),
            high_seq.to(device),
            task_ids,
        )["pred_fixation_map"].reshape(count, -1)
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if ior_radius > 0:
            prediction[visited] = 0.0
        totals = prediction.sum(dim=1, keepdim=True)
        if torch.any(totals <= 0.0):
            raise FloatingPointError("HAT produced an invalid sampling distribution")
        index = torch.multinomial(prediction / totals, 1, replacement=True)[:, 0]
        next_y = (index // pa.im_w).float() / float(pa.im_h)
        next_x = (index % pa.im_w).float() / float(pa.im_w)
        next_xy = torch.stack((next_x, next_y), dim=1)
        normalized = torch.cat((normalized, next_xy[:, None]), dim=1)
        outputs.append(torch.stack((next_y, next_x), dim=1))
        if ior_radius > 0:
            visited |= _wrapped_ior_mask(
                next_x, next_y, pa.im_h, pa.im_w, radius=ior_radius
            )
    return torch.stack(outputs, dim=1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def _sample_scanpaths(
    model: torch.nn.Module,
    image: torch.Tensor,
    count: int,
    pa: Any,
    device: torch.device,
    batch_paths: int,
    ior_radius: int,
) -> np.ndarray:
    model.eval()
    encoded = model.encode(image[None].to(device))
    chunks = []
    for begin in range(0, int(count), int(batch_paths)):
        current = min(int(batch_paths), int(count) - begin)
        chunks.append(
            _sample_chunk(
                model,
                encoded,
                current,
                pa,
                device,
                start_x=0.75,
                start_y=0.5,
                steps=20,
                ior_radius=ior_radius,
            )
        )
    result = np.concatenate(chunks, axis=0)
    if result.shape != (int(count), 20, 2) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid HAT scanpaths: {result.shape}")
    return result


def _load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    state = torch.load(checkpoint, map_location="cpu")
    pa = hat_config()
    model = build_model(pa)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, pa


def _generate(
    model: torch.nn.Module,
    pa: Any,
    images: dict[str, torch.Tensor],
    output: Path,
    count: int,
    device: torch.device,
    seed: int,
    batch_paths: int,
    ior_radius: int,
    force: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for image_id in sorted(images):
        path = output / f"{image_id}.npy"
        valid = False
        if path.exists() and not force:
            value = np.load(path)
            valid = value.shape == (int(count), 20, 2) and np.isfinite(value).all()
        if not valid:
            np.save(
                path,
                _sample_scanpaths(
                    model, images[image_id], count, pa, device, batch_paths, ior_radius
                ),
            )
        print(f"[HAT] {image_id}", flush=True)


def _predictions(scanpath_dir: Path, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        str(sample["image_id"]): second_sampled_scanpaths_to_cumulative_maps(
            np.load(scanpath_dir / f"{sample['image_id']}.npy"),
            horizons_s=sample["horizons_s"].numpy().tolist(),
            height=int(sample["target"].shape[-2]),
            width=int(sample["target"].shape[-1]),
            renderer="sphere_chord",
            sigma_deg=2.0,
            coordinate_order="yx",
        )
        for sample in samples
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-select and infer matched HAT checkpoints.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument(
        "--dataset-json",
        default="data/processed/kernel_v2_matched_scanpaths/hat/dataset.json",
    )
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--selection-paths", type=int, default=50)
    parser.add_argument("--final-paths", type=int, default=200)
    parser.add_argument("--batch-paths", type=int, default=25)
    parser.add_argument(
        "--ior-radius",
        type=int,
        default=0,
        help=(
            "Hard IOR radius in 512x256 pixels. The matched default is 0 because each "
            "generated point represents one second rather than one fixation event."
        ),
    )
    parser.add_argument("--required-final-step", type=int, default=40000)
    parser.add_argument("--run-prefix", default="kernel_v2_hat")
    parser.add_argument("--expected-val", type=int, default=9)
    parser.add_argument("--expected-gate", type=int, default=13)
    parser.add_argument("--expected-test", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed not in STANDARD_SEEDS:
        raise ValueError("Seed is outside the standard matched-baseline protocol")
    device = torch.device(args.device)
    train_run = (ROOT / args.train_run).resolve()
    checkpoints = sorted((train_run / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No HAT checkpoints in {train_run}")
    final_state = torch.load(checkpoints[-1], map_location="cpu")
    if int(final_state["step"]) != int(args.required_final_step):
        raise RuntimeError("HAT selection requires the complete formal training budget")
    dataset_path = (ROOT / args.dataset_json).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["_path"] = str(dataset_path)
    val_dataset = EarlyHorizonDataset(args.data_root, split="val", target_mode="correct")
    val_samples = [val_dataset[index] for index in range(len(val_dataset))]
    val_ids = {str(sample["image_id"]) for sample in val_samples}
    if val_ids != {str(row["image_id"]) for row in dataset["images"]["val"]}:
        raise ValueError("HAT val images do not match the standard saliency split")

    selection_records = []
    selection_root = train_run / f"selection_ior{int(args.ior_radius)}"
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu")
        step = int(state["step"])
        model, pa = _load_model(checkpoint, device)
        images = _images(dataset, "val", pa)
        scanpath_dir = selection_root / f"step_{step:06d}" / "scanpaths"
        _generate(
            model,
            pa,
            images,
            scanpath_dir,
            args.selection_paths,
            device,
            args.seed + 913,
            args.batch_paths,
            args.ior_radius,
        )
        scores, _ = _score(_predictions(scanpath_dir, val_samples), val_samples, (5.0, 10.0, 20.0))
        metric = _macro(scores, ("kld_sph", "sim_sph", "cc_sph"))
        selection_records.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                **metric,
            }
        )
        del model
        torch.cuda.empty_cache()
    selected = min(selection_records, key=lambda row: (float(row["kld_sph"]), int(row["step"])))
    selection = {
        "seed": args.seed,
        "criterion": "val early-macro KLD_sph over 0.5/1/3 s",
        "selection_paths_per_content": int(args.selection_paths),
        "records": selection_records,
        "selected": selected,
        "gate_used": False,
        "test_target_loads": 0,
        "initialization_semantics": "declared_start_history_token",
        "sampling_ior_radius": int(args.ior_radius),
        "sampling_ior_semantics": "disabled_for_second_sampled_gaze" if int(args.ior_radius) == 0 else "fixation_event_style_hard_ior",
    }
    (train_run / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    model, pa = _load_model(Path(str(selected["checkpoint"])), device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    final_splits = ("val", "test") if "standard_v3" in str(args.data_root) else ("val", "gate")
    for offset, split in enumerate(final_splits):
        images = _images(dataset, split, pa)
        expected = {
            "val": int(args.expected_val),
            "gate": int(args.expected_gate),
            "test": int(args.expected_test),
        }[split]
        if len(images) != expected:
            raise ValueError(f"HAT {split} count mismatch")
        final_run = ROOT / f"runs/{args.run_prefix}_seed{args.seed}_{split}"
        metadata_path = final_run / "inference_metadata.json"
        force = True
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            force = not (
                str(previous.get("checkpoint_sha256")) == str(selected["checkpoint_sha256"])
                and int(previous.get("scanpaths_per_content", -1)) == int(args.final_paths)
                and int(previous.get("sampling_ior_radius", -1)) == int(args.ior_radius)
            )
        _generate(
            model,
            pa,
            images,
            final_run / "scanpaths",
            args.final_paths,
            device,
            args.seed + 1901 + offset,
            args.batch_paths,
            args.ior_radius,
            force=force,
        )
        metadata = {
            "method": "HAT (ERP-adapted)",
            "seed": args.seed,
            "split": split,
            "selected_step": int(selected["step"]),
            "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "scanpaths_per_content": int(args.final_paths),
            "steps": 20,
            "coordinate_order": "normalized_yx",
            "start_mode": "declared",
            "start_pitch_deg": 0.0,
            "start_yaw_deg": 90.0,
            "initialization_semantics": "declared_start_history_token",
            "first_supervised_target": "first_one_second_gaze",
            "sampling_ior_radius": int(args.ior_radius),
            "sampling_ior_semantics": "disabled_for_second_sampled_gaze" if int(args.ior_radius) == 0 else "fixation_event_style_hard_ior",
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_count),
            "test_evaluated": bool(split == "test"),
            "test_used_for_selection": False,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"selected": selected, "parameters": parameter_count}, indent=2))


if __name__ == "__main__":
    main()
