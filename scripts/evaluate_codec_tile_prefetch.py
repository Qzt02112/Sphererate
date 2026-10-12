#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.codec_prefetch import allocate_upgrades, psnr_from_mse
from sphererate.evaluation.tile_prefetch import tile_probability_mass
from sphererate.models.registry import build_model


SEEDS = (20260710, 20260711, 20260712)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
BUDGETS = (5.0, 10.0, 15.0, 20.0)
METHODS = (
    "early_template",
    "tempsal_erp",
    "sphererate",
    "gt_attention_oracle",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rank_density(density: np.ndarray, area: np.ndarray) -> np.ndarray:
    scores = tile_probability_mass(
        density, area, tile_rows=8, tile_cols=16
    ).reshape(-1)
    ids = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((ids, -scores)).astype(np.int64)


def _rank_mass(mass: np.ndarray) -> np.ndarray:
    scores = np.asarray(mass, dtype=np.float64).reshape(-1)
    ids = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((ids, -scores)).astype(np.int64)


def _upsample_target(target: torch.Tensor, height: int, width: int) -> np.ndarray:
    resized = F.interpolate(
        target.view(1, 1, *target.shape),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy().astype(np.float32, copy=False)


def _codec_arrays(
    metadata: dict[str, Any], high_qp: int, low_qp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = int(metadata["height"]), int(metadata["width"])
    low_bytes, high_bytes = [], []
    low_error = np.empty((height, width), dtype=np.float32)
    high_error = np.empty((height, width), dtype=np.float32)
    bounds = []
    with Image.open(metadata["source_path"]) as original:
        original = original.convert("RGB")
        for tile in metadata["tiles"]:
            top, left, bottom, right = (int(value) for value in tile["bounds"])
            source = np.asarray(
                original.crop((left, top, right, bottom)), dtype=np.float32
            )
            with Image.open(tile["representations"][f"qp{low_qp}"]["decoded"]) as image:
                low = np.asarray(image.convert("RGB"), dtype=np.float32)
            with Image.open(tile["representations"][f"qp{high_qp}"]["decoded"]) as image:
                high = np.asarray(image.convert("RGB"), dtype=np.float32)
            low_error[top:bottom, left:right] = np.mean((source - low) ** 2, axis=-1)
            high_error[top:bottom, left:right] = np.mean((source - high) ** 2, axis=-1)
            low_bytes.append(int(tile["representations"][f"qp{low_qp}"]["bytes"]))
            high_bytes.append(int(tile["representations"][f"qp{high_qp}"]["bytes"]))
            bounds.append((top, left, bottom, right))
    return (
        np.asarray(low_bytes, dtype=np.int64),
        np.asarray(high_bytes, dtype=np.int64),
        low_error,
        high_error,
        np.asarray(bounds, dtype=np.int64),
    )


def _tile_weighted_errors(
    low_error: np.ndarray,
    high_error: np.ndarray,
    bounds: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    low, high = [], []
    for top, left, bottom, right in bounds:
        current = weights[top:bottom, left:right]
        low.append(float(np.sum(low_error[top:bottom, left:right] * current, dtype=np.float64)))
        high.append(float(np.sum(high_error[top:bottom, left:right] * current, dtype=np.float64)))
    return (
        np.asarray(low, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        float(np.sum(weights, dtype=np.float64)),
    )


def _mixed_psnr(
    low_error: np.ndarray,
    high_error: np.ndarray,
    selected: np.ndarray,
    denominator: float,
) -> float:
    numerator = float(low_error.sum() + (high_error[selected] - low_error[selected]).sum())
    return psnr_from_mse(numerator / max(float(denominator), 1e-12))


def _bootstrap(
    rows: list[dict[str, Any]],
    budget: float,
    baseline: str,
    metric: str,
    *,
    iterations: int = 10000,
) -> dict[str, float | int | str]:
    selected = [row for row in rows if float(row["budget_percent"]) == float(budget)]
    index = {
        (row["method"], row["content_id"], float(row["horizon_s"])): float(row[metric])
        for row in selected
    }
    contents = sorted({str(row["content_id"]) for row in selected})
    values = []
    lower_is_better = metric == "unvisited_upgrade_byte_fraction"
    for content in contents:
        differences = []
        for horizon in HORIZONS:
            candidate = index[("sphererate", content, horizon)]
            reference = index[(baseline, content, horizon)]
            differences.append(reference - candidate if lower_is_better else candidate - reference)
        values.append(float(np.mean(differences)))
    values_np = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260806)
    boot = values_np[
        rng.integers(0, len(values_np), size=(iterations, len(values_np)))
    ].mean(axis=1)
    return {
        "direction": "positive_means_sphererate_is_better",
        "mean": float(values_np.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "content_count": len(contents),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Real HEVC-Intra tile-prefetch evaluation.")
    parser.add_argument(
        "--data-root",
        default="data/processed/early_horizon_salient360_spherical2deg_standard_v3",
    )
    parser.add_argument(
        "--original-manifest",
        default="data/interim/salient360_codec_originals/manifest.jsonl",
    )
    parser.add_argument(
        "--codec-root", default="data/interim/salient360_tile_hevc_qp22_qp42"
    )
    parser.add_argument("--runs-root", default="runs/standard_split_v3")
    parser.add_argument("--output-dir", default="reports/codec_tile_prefetch_test")
    parser.add_argument("--high-qp", type=int, default=22)
    parser.add_argument("--low-qp", type=int, default=42)
    args = parser.parse_args()

    dataset = EarlyHorizonDataset(Path(args.data_root), split="test", tile_rows=8, tile_cols=16)
    originals = {
        row["image_id"]: row
        for row in _read_jsonl(Path(args.original_manifest))
        if row["split"] == "test"
    }
    if len(dataset) != 13 or len(originals) != 13:
        raise RuntimeError("Codec evaluation requires exactly the frozen 13-content test split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sphererate, tempsal = [], []
    for seed in SEEDS:
        sphere_state = torch.load(
            Path(args.runs_root)
            / f"v12c_hybrid_dwell_rate_seed{seed}/checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        sphererate.append(_model_from_checkpoint(sphere_state, device))
        temp_state = torch.load(
            Path(args.runs_root) / f"tempsal_erp_seed{seed}/checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build_model("semantic_tempsal_temporal", temp_state["config"]).to(device)
        model.load_state_dict(temp_state["model"])
        tempsal.append(model.eval())

    rows: list[dict[str, Any]] = []
    for content_index, item in enumerate(dataset, start=1):
        image_id = str(item["image_id"])
        metadata_path = Path(args.codec_root) / image_id / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        original = originals[image_id]
        if metadata["source_sha256"] != original["sha256"]:
            raise RuntimeError(f"Codec/source hash mismatch for {image_id}")
        low_bytes, high_bytes, low_error, high_error, bounds = _codec_arrays(
            metadata, args.high_qp, args.low_qp
        )
        all_low_bytes, all_high_bytes = int(low_bytes.sum()), int(high_bytes.sum())
        if all_high_bytes < all_low_bytes:
            raise RuntimeError(f"Invalid all-high/all-low byte order for {image_id}")
        height, width = low_error.shape
        latitude = np.pi / 2.0 - np.pi * (np.arange(height) + 0.5) / height
        ws_weights = np.broadcast_to(
            np.cos(latitude).astype(np.float32)[:, None], (height, width)
        )
        ws_low, ws_high, ws_denominator = _tile_weighted_errors(
            low_error, high_error, bounds, ws_weights
        )

        image = item["image"].unsqueeze(0).to(device)
        template = item["template"].unsqueeze(0).to(device)
        start = item["start_geometry"].unsqueeze(0).to(device)
        sphere_prediction = torch.stack(
            [model(image, template, start_geometry=start)["cumulative"][0] for model in sphererate]
        ).mean(0).float().cpu().numpy()
        tempsal_prediction = torch.stack(
            [model(image)["cumulative"][0] for model in tempsal]
        ).mean(0).float().cpu().numpy()
        area = item["area_weights"][0].numpy()

        for horizon_index, horizon in enumerate(HORIZONS):
            target = _upsample_target(item["target"][horizon_index], height, width)
            gaze_weights = target * ws_weights
            gaze_low, gaze_high, gaze_denominator = _tile_weighted_errors(
                low_error, high_error, bounds, gaze_weights
            )
            raw_dwell = item["raw_dwell_tiles"][horizon_index].numpy().reshape(-1)
            rankings = {
                "early_template": _rank_density(
                    item["template"][horizon_index].numpy(), area
                ),
                "tempsal_erp": _rank_density(tempsal_prediction[horizon_index], area),
                "sphererate": _rank_density(sphere_prediction[horizon_index], area),
                "gt_attention_oracle": _rank_mass(raw_dwell),
            }
            for budget_percent in BUDGETS:
                byte_cap = int(
                    math.floor(
                        all_low_bytes
                        + budget_percent / 100.0 * (all_high_bytes - all_low_bytes)
                    )
                )
                for method in METHODS:
                    selected = allocate_upgrades(
                        rankings[method], low_bytes, high_bytes, byte_cap
                    )
                    delta = high_bytes - low_bytes
                    actual_bytes = int(all_low_bytes + delta[selected].sum())
                    if actual_bytes > byte_cap:
                        raise RuntimeError("Byte allocator exceeded the common cap")
                    positive_upgrade = np.maximum(delta, 0)
                    used_upgrade_bytes = int(positive_upgrade[selected].sum())
                    unvisited = int(
                        positive_upgrade[selected & (raw_dwell <= 1e-12)].sum()
                    )
                    rows.append(
                        {
                            "method": method,
                            "content_id": str(item["canonical_content_id"]),
                            "image_id": image_id,
                            "horizon_s": horizon,
                            "budget_percent": budget_percent,
                            "byte_cap": byte_cap,
                            "actual_bytes": actual_bytes,
                            "bits_per_pixel": 8.0 * actual_bytes / float(height * width),
                            "budget_utilization": actual_bytes / float(byte_cap),
                            "upgraded_tile_count": int(selected.sum()),
                            "ws_psnr_db": _mixed_psnr(
                                ws_low, ws_high, selected, ws_denominator
                            ),
                            "gaze_psnr_db": _mixed_psnr(
                                gaze_low, gaze_high, selected, gaze_denominator
                            ),
                            "raw_dwell_coverage": float(raw_dwell[selected].sum()),
                            "unvisited_upgrade_bytes": unvisited,
                            "unvisited_upgrade_byte_fraction": (
                                unvisited / float(used_upgrade_bytes)
                                if used_upgrade_bytes
                                else 0.0
                            ),
                        }
                    )
        print(
            f"[{content_index}/{len(dataset)}] {image_id}: {width}x{height}, "
            f"{all_low_bytes}->{all_high_bytes} bytes",
            flush=True,
        )

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    report_metrics = (
        "actual_bytes",
        "bits_per_pixel",
        "budget_utilization",
        "ws_psnr_db",
        "gaze_psnr_db",
        "raw_dwell_coverage",
        "unvisited_upgrade_byte_fraction",
    )
    for method in METHODS:
        aggregate[method] = {}
        for budget in BUDGETS:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["budget_percent"] == budget
            ]
            aggregate[method][f"{budget:g}"] = {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in report_metrics
            }
    comparisons = {
        f"{budget:g}": {
            baseline: {
                metric: _bootstrap(rows, budget, baseline, metric)
                for metric in (
                    "gaze_psnr_db",
                    "ws_psnr_db",
                    "raw_dwell_coverage",
                    "unvisited_upgrade_byte_fraction",
                )
            }
            for baseline in ("early_template", "tempsal_erp")
        }
        for budget in BUDGETS
    }
    result = {
        "protocol": "real_independent_hevc_intra_tile_prefetch",
        "codec": {
            "implementation": "ffmpeg/libx265",
            "pixel_format": "yuv420p",
            "high_qp": args.high_qp,
            "low_qp": args.low_qp,
            "grid": "8x16",
        },
        "split": "standard_test_13_contents",
        "horizons_s": list(HORIZONS),
        "budgets_percent_of_low_to_high_byte_range": list(BUDGETS),
        "methods": list(METHODS),
        "aggregate_all_four_horizons": aggregate,
        "paired_sphererate_advantage": comparisons,
        "per_content_rows": len(rows),
        "target_usage": "quality/coverage evaluation and labeled GT-attention upper bound only",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per_content.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "early_template": "Early template",
        "tempsal_erp": "TempSAL-ERP",
        "sphererate": "SphereRate",
        "gt_attention_oracle": "GT-attention oracle",
    }
    lines = [
        "# 真实 HEVC-Intra tile 传输实验（standard test）",
        "",
        "每个原始分辨率 ERP 被切成 8×16 个独立可解码 HEVC-Intra tiles。QP42 为基础层，模型按注意力排序将 tiles 升级至 QP22；每张图、每个预算下所有方法共享同一个实际字节上限。四个时刻均计入 early-horizon 汇总。",
        "",
    ]
    for budget in BUDGETS:
        lines.extend(
            [
                f"## Upgrade budget {budget:g}%",
                "",
                "| Method | bpp | WS-PSNR↑ | Gaze-PSNR↑ | Dwell coverage↑ | Unvisited HQ bytes↓ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            value = aggregate[method][f"{budget:g}"]
            lines.append(
                f"| {labels[method]} | {value['bits_per_pixel']:.4f} | "
                f"{value['ws_psnr_db']:.3f} | {value['gaze_psnr_db']:.3f} | "
                f"{value['raw_dwell_coverage']:.4f} | "
                f"{100.0 * value['unvisited_upgrade_byte_fraction']:.2f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "GT-attention oracle 只按真实 dwell mass 排序，表示注意力已知上界，并非联合考虑纹理复杂度的 rate-distortion oracle。Gaze-PSNR 中 GT 仅作为评价权重，不参与三个可部署方法的 tile 排序。paired bootstrap 以 13 个内容为单位，详见 `results.json`。",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
