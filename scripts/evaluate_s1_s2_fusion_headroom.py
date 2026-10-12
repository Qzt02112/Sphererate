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

from sphererate.evaluation.metrics import kld
from sphererate.evaluation.semantic_reranking import geometric_semantic_fusion
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned
from scripts.evaluate_semantic_reranking_oracle import (
    _aggregate,
    _prediction_row,
    _samples,
)


def _load_fixations(sample: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    with np.load(str(sample["fixations_path"])) as data:
        fixations = np.asarray(data["fixations"], dtype=np.float32)
    return fixations, np.unique(fixations[:, 3]).astype(int).tolist()


def _fit(
    samples: list[dict[str, Any]],
    base: dict[str, np.ndarray],
    concentration: dict[str, np.ndarray],
    *,
    horizons: tuple[float, ...],
    grid: tuple[float, ...],
) -> tuple[dict[float, float], dict[str, Any]]:
    selected: dict[float, float] = {}
    traces: dict[str, Any] = {}
    for index, horizon in enumerate(horizons):
        prepared = []
        for sample in samples:
            image_id = str(sample["image_id"])
            area = sample["area_weights"].numpy()[0]
            target = sample["target"].numpy()[index]
            fixations, observers = _load_fixations(sample)
            prepared.append((image_id, area, target, fixations, observers))
        base_kld = float(
            np.mean([kld(base[key][index], target, area) for key, area, target, _, _ in prepared])
        )
        candidates = []
        for beta in grid:
            klds, coverages = [], {5.0: [], 10.0: [], 20.0: []}
            for image_id, area, target, fixations, observers in prepared:
                prediction = geometric_semantic_fusion(
                    base[image_id][index], concentration[image_id][index], area, alpha=beta
                )
                klds.append(kld(prediction, target, area))
                for budget in coverages:
                    tiles = rank_tiles(
                        prediction, area, tile_rows=8, tile_cols=16, budget_percent=budget
                    )
                    coverages[budget].append(
                        gaze_mass_coverage(
                            fixations,
                            observer_ids=observers,
                            horizon_s=horizon,
                            selected_tile_ids=tiles,
                            tile_rows=8,
                            tile_cols=16,
                        )
                    )
            row = {
                "beta": beta,
                "kld_sph": float(np.mean(klds)),
                **{f"coverage_{budget:g}": float(np.mean(values)) for budget, values in coverages.items()},
            }
            row["coverage_selection_mean"] = 0.5 * (row["coverage_5"] + row["coverage_10"])
            row["passes_kld_guard"] = row["kld_sph"] <= base_kld + 0.02
            candidates.append(row)
        feasible = [row for row in candidates if row["passes_kld_guard"]]
        choice = max(
            feasible,
            key=lambda row: (row["coverage_selection_mean"], -row["kld_sph"], -row["beta"]),
        )
        selected[horizon] = float(choice["beta"])
        traces[f"{horizon:g}"] = {"base_kld": base_kld, "candidates": candidates}
    return selected, traces


def _evaluate(
    samples: list[dict[str, Any]],
    base: dict[str, np.ndarray],
    concentration: dict[str, np.ndarray],
    betas: dict[float, float],
    *,
    horizons: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        image_id = str(sample["image_id"])
        area = sample["area_weights"].numpy()[0]
        fixations, observers = _load_fixations(sample)
        for index, horizon in enumerate(horizons):
            prediction = geometric_semantic_fusion(
                base[image_id][index],
                concentration[image_id][index],
                area,
                alpha=betas[horizon],
            )
            rows.append(
                _prediction_row(
                    prediction,
                    sample,
                    horizon_index=index,
                    horizon=horizon,
                    budgets=(5.0, 10.0, 20.0),
                    fixations=fixations,
                    observers=observers,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="No-training S1/S2 fusion headroom diagnostic.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="reports/semantic_reranking_s1_s2_headroom")
    args = parser.parse_args()
    horizons = (0.5, 1.0, 3.0, 5.0)
    primary = {0.5, 1.0, 3.0}
    root = ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2"
    train_samples = _samples(root, "train")
    val_samples = _samples(root, "val")
    device = torch.device(args.device)
    s1_path = ROOT / "runs/semantic_reranking_s1_seed20260711/checkpoints/best.pt"
    s2_path = ROOT / "runs/semantic_reranking_s2_matched_anchor_seed20260711/checkpoints/last.pt"
    s1_train, _ = _predict_learned(s1_path, train_samples, device)
    s2_train, _ = _predict_learned(s2_path, train_samples, device)
    betas, traces = _fit(
        train_samples,
        s2_train,
        s1_train,
        horizons=horizons,
        grid=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0),
    )
    s1_val, _ = _predict_learned(s1_path, val_samples, device)
    s2_val, _ = _predict_learned(s2_path, val_samples, device)
    rows = _evaluate(val_samples, s2_val, s1_val, betas, horizons=horizons)
    fused = _aggregate(rows, horizons=horizons, primary=primary, budgets=(5.0, 10.0, 20.0))
    s0 = json.loads((ROOT / "reports/semantic_reranking_s0/results.json").read_text())
    strongest = {
        budget: max(
            s0["methods"]["spherehorizon"]["early_macro"][f"coverage_{budget:g}"],
            s0["methods"]["tempsal"]["early_macro"][f"coverage_{budget:g}"],
        )
        for budget in (5.0, 10.0)
    }
    current = fused["early_macro"]
    passed = (
        current["kld_sph"] <= 1.10999
        and current["coverage_5"] > strongest[5.0]
        and current["coverage_10"] > strongest[10.0]
    )
    result = {
        "decision": "PASS_AUTHORIZE_SHARED_TWO_HEAD" if passed else "FAIL_STOP_ARCHITECTURE_EXPANSION",
        "protocol": {
            "beta_selection": "Salient360 train only",
            "evaluation": "Salient360 val",
            "new_training": False,
            "inference_efficiency": "diagnostic two-network fusion; not a proposed method",
            "gate_test_target_loads": 0,
        },
        "betas": {f"{key:g}": value for key, value in betas.items()},
        "train_traces": traces,
        "fused": fused,
        "references": {
            "spherehorizon": s0["methods"]["spherehorizon"],
            "tempsal": s0["methods"]["tempsal"],
        },
    }
    out = ensure_dir(ROOT / args.output)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# S1/S2 No-training Fusion Headroom",
        "",
        f"Decision: **{result['decision']}**. This is a diagnostic two-network fusion, not a proposed model.",
        "",
        f"Train-selected beta (S1 concentration contribution): {result['betas']}.",
        "",
        "| Method | KLD ↓ | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, method in (
        ("S1/S2 diagnostic fusion", fused),
        ("SphereHorizon", result["references"]["spherehorizon"]),
        ("TempSAL", result["references"]["tempsal"]),
    ):
        row = method["early_macro"]
        summary.append(
            f"| {name} | {row['kld_sph']:.4f} | {row['coverage_5']:.4f} | "
            f"{row['coverage_10']:.4f} | {row['coverage_20']:.4f} |"
        )
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "decision": result["decision"], "betas": result["betas"]}, indent=2))


if __name__ == "__main__":
    main()
