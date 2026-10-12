#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.losses import weighted_cc, weighted_kld, weighted_sim


SEEDS = (20260710, 20260711, 20260712)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
BUDGETS = (5.0, 10.0, 15.0, 20.0)
METRICS = (
    "kld_sph",
    "sim_sph",
    "cc_sph",
    "coverage_5",
    "coverage_10",
    "coverage_15",
    "coverage_20",
)
VARIANTS = {
    "full": "v12c_hybrid_dwell_rate_seed{seed}",
    "no_rate": "v12c_no_rate_seed{seed}",
    "no_start": "v12c_no_start_seed{seed}",
    "no_tile_loss": "v12c_no_tile_loss_seed{seed}",
    # Isolates L_cov only: L_tile stays on, the budget-limited coverage
    # objective is switched off.
    "no_cov": "v12c_no_cov_seed{seed}",
    # Isolates the temporal interval-residual branch: rho is forced to zero,
    # so with the semantic anchor frozen the only trainable content parameters
    # disappear and per-scale alpha/beta/offsets are all that remain.
    "no_residual": "v12c_no_residual_seed{seed}",
    # Isolates L_tile only: L_cov stays on, the raw-tile KL objective is
    # switched off.  With no_cov and no_tile_loss this completes the 2x2
    # over the two auxiliary tile losses.
    "no_tile_only": "v12c_no_tile_only_seed{seed}",
}


def _checkpoint_paths(run_root: Path) -> dict[str, list[Path]]:
    paths = {
        name: [
            run_root / pattern.format(seed=seed) / "checkpoints" / "best.pt"
            for seed in SEEDS
        ]
        for name, pattern in VARIANTS.items()
    }
    missing = [str(path) for current in paths.values() for path in current if not path.is_file()]
    if missing:
        raise FileNotFoundError("All checkpoints must exist before test evaluation: " + repr(missing))
    return paths


def _map_metrics(
    prediction: torch.Tensor, target: torch.Tensor, area: torch.Tensor
) -> dict[str, float]:
    pred = prediction.view(1, 1, *prediction.shape)
    gt = target.view(1, 1, *target.shape)
    weights = area.view(1, 1, *area.shape)
    return {
        "kld_sph": float(weighted_kld(pred, gt, weights)),
        "sim_sph": float(weighted_sim(pred, gt, weights)),
        "cc_sph": float(weighted_cc(pred, gt, weights)),
    }


def _coverage(
    prediction: np.ndarray,
    area: np.ndarray,
    fixations: np.ndarray,
    observers: list[int],
    horizon: float,
) -> dict[str, float]:
    return {
        f"coverage_{budget:g}": gaze_mass_coverage(
            fixations,
            observer_ids=observers,
            horizon_s=horizon,
            selected_tile_ids=rank_tiles(
                prediction,
                area,
                tile_rows=8,
                tile_cols=16,
                budget_percent=budget,
            ),
            tile_rows=8,
            tile_cols=16,
        )
        for budget in BUDGETS
    }


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    selected = [row for row in rows if row["variant"] == variant]
    return {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}


def _per_horizon(
    rows: list[dict[str, Any]], variant: str, horizon: float
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["variant"] == variant and float(row["horizon_s"]) == float(horizon)
    ]
    return {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}


def _paired_bootstrap(
    rows: list[dict[str, Any]], ablation: str, metric: str, iterations: int = 10000
) -> dict[str, float | int | str]:
    index = {
        (str(row["variant"]), str(row["content_id"]), float(row["horizon_s"])): float(
            row[metric]
        )
        for row in rows
    }
    contents = sorted({str(row["content_id"]) for row in rows})
    differences = []
    for content in contents:
        current = []
        for horizon in HORIZONS:
            full = index[("full", content, horizon)]
            ablated = index[(ablation, content, horizon)]
            current.append(ablated - full if metric == "kld_sph" else full - ablated)
        differences.append(float(np.mean(current)))
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20260806)
    sampled = values[
        rng.integers(0, len(values), size=(int(iterations), len(values)))
    ].mean(axis=1)
    return {
        "direction": "positive_means_full_is_better",
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "content_count": int(len(values)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matched SphereRate mechanisms.")
    parser.add_argument(
        "--data-root",
        default="data/processed/early_horizon_salient360_spherical2deg_standard_v3",
    )
    parser.add_argument("--run-root", default="runs/standard_split_v3")
    parser.add_argument("--output-dir", default="reports/sphererate_ablation_test")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    checkpoint_paths = _checkpoint_paths(run_root)
    dataset = EarlyHorizonDataset(Path(args.data_root), split="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    checkpoint_audit: dict[str, list[dict[str, Any]]] = {}
    consistency: dict[str, dict[str, int | bool]] = {}

    for variant, paths in checkpoint_paths.items():
        checkpoints = [
            torch.load(path, map_location="cpu", weights_only=False) for path in paths
        ]
        models = [_model_from_checkpoint(checkpoint, device) for checkpoint in checkpoints]
        checkpoint_audit[variant] = [
            {
                "seed": seed,
                "path": str(path),
                "epoch": int(checkpoint["epoch"]),
                "parameters": checkpoint["parameters"],
                "selection": checkpoint["selection"],
                "semantic_anchor_sha256": checkpoint["initialization"]["sha256"],
            }
            for seed, path, checkpoint in zip(SEEDS, paths, checkpoints)
        ]
        negative_increments = 0
        checked_increments = 0
        for item in dataset:
            outputs = [
                model(
                    item["image"].unsqueeze(0).to(device),
                    item["template"].unsqueeze(0).to(device),
                    start_geometry=item["start_geometry"].unsqueeze(0).to(device),
                )
                for model in models
            ]
            for output in outputs:
                increments = output["raw_cumulative_mass"][:, 1:] - output[
                    "raw_cumulative_mass"
                ][:, :-1]
                negative_increments += int((increments < -1e-7).sum().item())
                checked_increments += int(increments.numel())
            fixations = np.load(str(item["fixations_path"]))["fixations"].astype(np.float32)
            observers = np.unique(fixations[:, 3]).astype(int).tolist()
            area_t = item["area_weights"][0]
            area_np = area_t.numpy()
            for horizon_index, horizon in enumerate(HORIZONS):
                seed_metrics = []
                for output in outputs:
                    prediction = output["cumulative"][0, horizon_index].float().cpu()
                    current = _map_metrics(
                        prediction, item["target"][horizon_index], area_t
                    )
                    current.update(
                        _coverage(
                            prediction.numpy(), area_np, fixations, observers, horizon
                        )
                    )
                    seed_metrics.append(current)
                rows.append(
                    {
                        "variant": variant,
                        "content_id": str(item["canonical_content_id"]),
                        "image_id": str(item["image_id"]),
                        "horizon_s": horizon,
                        **{
                            metric: float(np.mean([value[metric] for value in seed_metrics]))
                            for metric in METRICS
                        },
                    }
                )
        consistency[variant] = {
            "rate_accumulation_used_for_prediction": variant != "no_rate",
            "negative_raw_mass_increments": negative_increments,
            "checked_raw_mass_increments": checked_increments,
        }
        del models

    total_counts = {
        audit["parameters"]["total"]
        for current in checkpoint_audit.values()
        for audit in current
    }
    trainable_counts = {
        audit["parameters"]["trainable"]
        for current in checkpoint_audit.values()
        for audit in current
    }
    if len(total_counts) != 1 or len(trainable_counts) != 1:
        raise RuntimeError("Mechanism ablations changed parameter counts")

    macro = {variant: _aggregate(rows, variant) for variant in VARIANTS}
    per_horizon = {
        variant: {
            f"{horizon:g}": _per_horizon(rows, variant, horizon)
            for horizon in HORIZONS
        }
        for variant in VARIANTS
    }
    comparisons = {
        ablation: {
            metric: _paired_bootstrap(rows, ablation, metric) for metric in METRICS
        }
        for ablation in (
            "no_rate",
            "no_start",
            "no_tile_loss",
            "no_cov",
            "no_tile_only",
            "no_residual",
        )
    }
    result = {
        "protocol": "matched_three_seed_standard_test_mechanism_ablation",
        "split": "test",
        "horizons_s": list(HORIZONS),
        "macro_definition": "unweighted mean over 13 contents and all four early horizons",
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "parameter_count": int(next(iter(total_counts))),
        "trainable_parameter_count": int(next(iter(trainable_counts))),
        "macro": macro,
        "per_horizon": per_horizon,
        "paired_full_advantage": comparisons,
        "cumulative_consistency": consistency,
        "checkpoints": checkpoint_audit,
        "test_contents": len(dataset),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per_content.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "full": "SphereRate (full)",
        "no_rate": "w/o rate accumulation",
        "no_start": "w/o declared-start cost",
        "no_tile_loss": "w/o tile losses",
        "no_cov": "w/o coverage loss",
        "no_tile_only": "w/o tile KL loss",
        "no_residual": "w/o temporal residual",
    }
    lines = [
        "# SphereRate 机制消融（standard test）",
        "",
        "四个时刻（0.5/1/3/5 s）均视为 early horizon；每个单元先在三随机种子间取均值，再以内容为统计单位。所有变体参数量与训练预算完全一致。",
        "",
        "| Variant | KLD↓ | SIM↑ | CC↑ | Cov@5↑ | Cov@10↑ | Cov@15↑ | Cov@20↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = macro[variant]
        lines.append(
            f"| {labels[variant]} | {value['kld_sph']:.4f} | {value['sim_sph']:.4f} | "
            f"{value['cc_sph']:.4f} | {value['coverage_5']:.4f} | "
            f"{value['coverage_10']:.4f} | {value['coverage_15']:.4f} | "
            f"{value['coverage_20']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定原则",
            "",
            "paired bootstrap 的差值统一定义为正值代表 full 更好。95% CI 跨 0 的机制只记为中性证据，不作因果性贡献声明。详细的逐时刻结果与区间见 `results.json`。",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
