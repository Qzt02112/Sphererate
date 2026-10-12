#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.evaluation.semantic_reranking import deterministic_derangement
from sphererate.train import load_model_for_inference
from sphererate.utils import ensure_dir
from scripts.evaluate_semantic_reranking_oracle import (
    _aggregate,
    _evaluate_fusion,
    _fit_alphas,
    _paired_bootstrap,
    _samples,
)


@torch.no_grad()
def _predict_static(
    checkpoint: Path,
    samples: list[dict[str, Any]],
    *,
    device_name: str,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    model, _, device = load_model_for_inference(checkpoint, device_name=device_name)
    predictions: dict[str, np.ndarray] = {}
    for start in range(0, len(samples), batch_size):
        current = samples[start : start + batch_size]
        images = torch.stack([sample["image"] for sample in current]).to(device)
        maps = model(images)["final"].detach().float().cpu().numpy()
        for index, sample in enumerate(current):
            predictions[str(sample["image_id"])] = maps[index, 0].astype(np.float64)
    return predictions


def _wrong_image_maps(
    samples: list[dict[str, Any]],
    maps: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    permutation = deterministic_derangement(len(samples), seed=seed, key="matched-static-val")
    return {
        str(sample["image_id"]): maps[str(samples[int(permutation[index])]["image_id"])]
        for index, sample in enumerate(samples)
    }


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Matched Static Semantic-Anchor Diagnostic",
        "",
        "The static model is trained on the same Salient360 train split. Fusion coefficients are selected on train only; all scores below are on val. No new training and no gate/test access are used.",
        "",
        "## Train-selected coefficients",
        "",
        "| 0.5 s | 1 s | 3 s | 5 s |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(f"{result['alphas'][key]:.2f}" for key in ("0.5", "1", "3", "5")) + " |",
        "",
        "## Validation results",
        "",
        "| Method | KLD ↓ | CC ↑ | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("template", "matched_static_fusion", "wrong_image_fusion", "spherehorizon", "tempsal"):
        row = result["methods"][method]["early_macro"]
        lines.append(
            f"| {method} | {row['kld_sph']:.4f} | {row['cc_sph']:.4f} | "
            f"{row['coverage_5']:.4f} | {row['coverage_10']:.4f} | {row['coverage_20']:.4f} |"
        )
    lines.extend(["", "## Attribution at 3 s", ""])
    for label, metrics in result["comparisons_3s"].items():
        lines.append(
            f"- {label}: Cov@5 {metrics['coverage_5']['mean']:+.4f} "
            f"[{metrics['coverage_5']['ci95_low']:+.4f}, {metrics['coverage_5']['ci95_high']:+.4f}]; "
            f"Cov@10 {metrics['coverage_10']['mean']:+.4f} "
            f"[{metrics['coverage_10']['ci95_low']:+.4f}, {metrics['coverage_10']['ci95_high']:+.4f}]."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a matched static map as a semantic anchor.")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--checkpoint", default="runs/kernel_v2_static_final_seed20260711/checkpoints/best.pt")
    parser.add_argument("--output", default="reports/matched_static_semantic_anchor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    horizons = (0.5, 1.0, 3.0, 5.0)
    primary = {0.5, 1.0, 3.0}
    budgets = (5.0, 10.0, 20.0)
    alpha_grid = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
    train_samples = _samples(ROOT / args.data_root, "train")
    val_samples = _samples(ROOT / args.data_root, "val")
    checkpoint = ROOT / args.checkpoint
    train_maps = _predict_static(checkpoint, train_samples, device_name=args.device)
    val_maps = _predict_static(checkpoint, val_samples, device_name=args.device)
    alphas, traces = _fit_alphas(
        train_samples,
        train_maps,
        horizons=horizons,
        budgets=budgets,
        selection_budgets=(5.0, 10.0),
        alpha_grid=alpha_grid,
        kld_guard=0.02,
    )
    zero = {horizon: 0.0 for horizon in horizons}
    rows = {
        "template": _evaluate_fusion(
            val_samples, val_maps, zero, horizons=horizons, budgets=budgets
        ),
        "matched_static_fusion": _evaluate_fusion(
            val_samples, val_maps, alphas, horizons=horizons, budgets=budgets
        ),
        "wrong_image_fusion": _evaluate_fusion(
            val_samples,
            _wrong_image_maps(val_samples, val_maps, seed=args.seed),
            alphas,
            horizons=horizons,
            budgets=budgets,
        ),
    }
    s0 = json.loads((ROOT / "reports/semantic_reranking_s0/results.json").read_text())
    methods = {
        name: _aggregate(current, horizons=horizons, primary=primary, budgets=budgets)
        for name, current in rows.items()
    }
    methods["spherehorizon"] = s0["methods"]["spherehorizon"]
    methods["tempsal"] = s0["methods"]["tempsal"]
    comparisons = {}
    for label, right in (
        ("matched-minus-template", "template"),
        ("matched-minus-wrong", "wrong_image_fusion"),
    ):
        comparisons[label] = {
            metric: _paired_bootstrap(
                rows["matched_static_fusion"],
                rows[right],
                horizon=3.0,
                metric=metric,
                iterations=10000,
                seed=args.seed,
            )
            for metric in ("coverage_5", "coverage_10")
        }
    result = {
        "protocol": {
            "alpha_selection": "Salient360 train only",
            "evaluation": "Salient360 val",
            "checkpoint": str(checkpoint),
            "new_training": False,
            "gate_test_target_loads": 0,
        },
        "alphas": {f"{key:g}": value for key, value in alphas.items()},
        "alpha_traces": traces,
        "methods": methods,
        "comparisons_3s": comparisons,
    }
    out = ensure_dir(ROOT / args.output)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(json.dumps({"output": str(out), "alphas": result["alphas"]}, indent=2))


if __name__ == "__main__":
    main()
