#!/usr/bin/env python3
"""Export one coherent set of real SphereHorizon intermediates for Fig. 1.

This script performs inference only.  It uses the frozen central checkpoint and
the already-fixed P10 qualitative content; it does not train, select, or tune a
model.  All ERP-domain assets are exported at the native 256x512 (1:2) shape.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_eh3_gate import _build_model_from_state
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import rank_tiles


CONTENT_ID = "P10"
HORIZON_INDEX = 2  # 3 s, fixed before this figure pass
SEED = 20260711
CHECKPOINT = ROOT / f"runs/kernel_v2_candidate_correct_seed{SEED}/checkpoints/best.pt"
CACHE = ROOT / f"runs/kernel_v2_candidate_correct_seed{SEED}/predictions/gate/{CONTENT_ID}.npz"
OUT = ROOT / "reports/spherehorizon_overview/assets/actual_p10_seed20260711"


def _rgb8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value) * 255.0), 0, 255).astype(np.uint8)


def _density_rgb(value: np.ndarray) -> np.ndarray:
    """Per-panel linear display normalization with a perceptually uniform map."""
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    vmax = float(np.percentile(array, 99.7))
    scaled = np.clip(array / max(vmax, 1e-12), 0.0, 1.0)
    # Compact approximation to the perceptually ordered magma palette.  Keeping
    # the stops local avoids importing a binary Matplotlib build into inference.
    stops = np.asarray(
        [
            [0, 0, 4],
            [59, 15, 112],
            [140, 41, 129],
            [222, 73, 104],
            [254, 159, 109],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )
    position = scaled * float(len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    alpha = (position - lower)[..., None]
    mapped = stops[lower] * (1.0 - alpha) + stops[upper] * alpha
    return np.clip(np.rint(mapped), 0, 255).astype(np.uint8)


def _residual_rgb(value: np.ndarray) -> np.ndarray:
    """Signed teal-white-orange rendering of the actual residual logits."""
    array = np.asarray(value, dtype=np.float32)
    array = array - float(np.median(array))
    bound = float(np.percentile(np.abs(array), 99.0))
    scaled = np.clip(array / max(bound, 1e-12), -1.0, 1.0)
    teal = np.asarray([37, 135, 127], dtype=np.float32)
    white = np.asarray([248, 248, 246], dtype=np.float32)
    orange = np.asarray([218, 132, 42], dtype=np.float32)
    rgb = np.empty((*scaled.shape, 3), dtype=np.float32)
    positive = scaled >= 0
    alpha_pos = scaled[positive, None]
    alpha_neg = (-scaled[~positive])[:, None]
    rgb[positive] = white * (1.0 - alpha_pos) + orange * alpha_pos
    rgb[~positive] = white * (1.0 - alpha_neg) + teal * alpha_neg
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path)


@torch.no_grad()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = EarlyHorizonDataset(
        ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    sample = next(dataset[index] for index in range(len(dataset)) if str(dataset[index]["image_id"]) == CONTENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location=device)
    model = _build_model_from_state(state, device)
    output = model(
        sample["image"].unsqueeze(0).to(device),
        sample["template"].unsqueeze(0).to(device),
    )
    prediction = output["cumulative"][0].cpu().numpy().astype(np.float32)
    residual = output["residual_logits"][0].cpu().numpy().astype(np.float32)
    gates = output["gates"][0].cpu().numpy().astype(np.float32)
    template = sample["template"].numpy().astype(np.float32)
    rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
    area = sample["area_weights"].numpy()[0].astype(np.float32)
    horizons = sample["horizons_s"].numpy().astype(np.float32)

    with np.load(CACHE) as cached:
        cache_prediction = np.asarray(cached["cumulative"], dtype=np.float32)
    max_cache_error = float(np.max(np.abs(prediction - cache_prediction)))
    if max_cache_error > 5e-7:
        raise RuntimeError(f"Fresh inference does not match frozen cache: {max_cache_error}")

    height, width = prediction.shape[-2:]
    if width != 2 * height:
        raise ValueError(f"Expected 1:2 ERP arrays, got {height}x{width}")

    # The frozen model uses x_ref=0.75.  Fig. 1 re-expresses the same arrays in
    # a reader-friendly centered gauge: declared start x=0.75 -> x_ref=0.5.
    center_shift = -(width // 4)
    aligned_rgb = np.roll(rgb, shift=center_shift, axis=1)
    centered_template = np.roll(template, shift=center_shift, axis=-1)
    centered_residual = np.roll(residual, shift=center_shift, axis=-1)
    centered_prediction = np.roll(prediction, shift=center_shift, axis=-1)
    restored_prediction = np.roll(centered_prediction, shift=-center_shift, axis=-1)
    restore_error = float(np.max(np.abs(restored_prediction - prediction)))

    _save_png(OUT / "erp_input.png", _rgb8(rgb))
    _save_png(OUT / "erp_aligned_center.png", _rgb8(aligned_rgb))
    for index in range(len(horizons)):
        _save_png(OUT / f"template_t{index + 1}.png", _density_rgb(centered_template[index]))
    _save_png(OUT / "residual_t3.png", _residual_rgb(centered_residual[HORIZON_INDEX]))
    _save_png(OUT / "fused_center_t3.png", _density_rgb(centered_prediction[HORIZON_INDEX]))
    _save_png(OUT / "early_map_restored_t3.png", _density_rgb(restored_prediction[HORIZON_INDEX]))

    selected = rank_tiles(
        restored_prediction[HORIZON_INDEX],
        area,
        tile_rows=8,
        tile_cols=16,
        budget_percent=10.0,
    )
    selected_mask = np.zeros((8, 16), dtype=np.uint8)
    selected_mask.ravel()[selected] = 1
    np.savez_compressed(
        OUT / "actuals.npz",
        rgb=rgb,
        aligned_rgb=aligned_rgb,
        template_center=centered_template,
        residual_center=centered_residual,
        prediction_center=centered_prediction,
        prediction_restored=restored_prediction,
        selected_tile_mask=selected_mask,
        horizons_s=horizons,
        gates=gates,
        area_weights=area,
    )
    provenance = {
        "content_id": CONTENT_ID,
        "split": "historically_seen_gate",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "seed": SEED,
        "checkpoint_epoch": int(state["epoch"]),
        "horizon_index_for_residual_output_tiles": HORIZON_INDEX,
        "horizon_s": float(horizons[HORIZON_INDEX]),
        "native_shape_hw": [height, width],
        "erp_aspect_hw": "1:2",
        "model_reference_x_norm": 0.75,
        "figure_reference_x_norm": 0.5,
        "figure_center_shift_px": center_shift,
        "fresh_vs_frozen_cache_max_abs_error": max_cache_error,
        "inverse_restore_max_abs_error": restore_error,
        "selected_tiles": selected.astype(int).tolist(),
        "selected_tile_count": int(selected.size),
        "display": {
            "density": "raw map, per-panel 99.7th-percentile linear normalization, magma",
            "residual": "raw residual logits, median-centered, symmetric p99 teal-white-orange",
            "smoothing": "none",
        },
        "training_or_selection_performed": False,
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
