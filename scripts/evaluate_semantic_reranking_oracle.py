#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.config import load_config
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.semantic_reranking import (
    deterministic_derangement,
    geometric_semantic_fusion,
)
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned


METRICS = ("kld_sph", "sim_sph", "cc_sph")


def _samples(root: Path, split: str) -> list[dict[str, Any]]:
    dataset = EarlyHorizonDataset(root, split=split, target_mode="correct")
    return [dataset[index] for index in range(len(dataset))]


def _load_fixations(sample: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    with np.load(str(sample["fixations_path"])) as data:
        fixations = np.asarray(data["fixations"], dtype=np.float32)
    observers = np.unique(fixations[:, 3]).astype(int).tolist()
    return fixations, observers


def _semantic_maps(
    samples: list[dict[str, Any]],
    *,
    source: str,
    split: str,
    deepgaze_root: Path,
    wrong_seed: int = 20260714,
) -> dict[str, np.ndarray]:
    if source == "gt_final_oracle":
        return {
            str(sample["image_id"]): np.asarray(sample["final"].numpy()[0], dtype=np.float64)
            for sample in samples
        }
    if source == "wrong_image_gt_final":
        permutation = deterministic_derangement(len(samples), seed=wrong_seed, key=f"semantic-s0:{split}")
        return {
            str(sample["image_id"]): np.asarray(samples[int(permutation[index])]["final"].numpy()[0], dtype=np.float64)
            for index, sample in enumerate(samples)
        }
    if source == "deepgaze_raw":
        maps = {}
        for sample in samples:
            image_id = str(sample["image_id"])
            path = deepgaze_root / split / "maps" / f"{image_id}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            maps[image_id] = np.asarray(np.load(path), dtype=np.float64)
        return maps
    raise ValueError(f"Unknown semantic source: {source}")


def _prediction_row(
    prediction: np.ndarray,
    sample: dict[str, Any],
    *,
    horizon_index: int,
    horizon: float,
    budgets: tuple[float, ...],
    fixations: np.ndarray,
    observers: list[int],
) -> dict[str, Any]:
    target = np.asarray(sample["target"].numpy()[horizon_index], dtype=np.float64)
    area = np.asarray(sample["area_weights"].numpy()[0], dtype=np.float64)
    row: dict[str, Any] = {
        "content_id": str(sample["canonical_content_id"]),
        "image_id": str(sample["image_id"]),
        "horizon_s": float(horizon),
        "kld_sph": kld(prediction, target, area),
        "sim_sph": sim(prediction, target, area),
        "cc_sph": cc(prediction, target, area),
    }
    for budget in budgets:
        selected = rank_tiles(
            prediction,
            area,
            tile_rows=8,
            tile_cols=16,
            budget_percent=float(budget),
        )
        row[f"coverage_{budget:g}"] = gaze_mass_coverage(
            fixations,
            observer_ids=observers,
            horizon_s=float(horizon),
            selected_tile_ids=selected,
            tile_rows=8,
            tile_cols=16,
        )
    return row


def _evaluate_fusion(
    samples: list[dict[str, Any]],
    semantic: dict[str, np.ndarray],
    alphas: dict[float, float],
    *,
    horizons: tuple[float, ...],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        image_id = str(sample["image_id"])
        template = np.asarray(sample["template"].numpy(), dtype=np.float64)
        area = np.asarray(sample["area_weights"].numpy()[0], dtype=np.float64)
        fixations, observers = _load_fixations(sample)
        for index, horizon in enumerate(horizons):
            prediction = geometric_semantic_fusion(
                template[index], semantic[image_id], area, alpha=alphas[float(horizon)]
            )
            rows.append(
                _prediction_row(
                    prediction,
                    sample,
                    horizon_index=index,
                    horizon=horizon,
                    budgets=budgets,
                    fixations=fixations,
                    observers=observers,
                )
            )
    return rows


def _evaluate_checkpoints(
    samples: list[dict[str, Any]],
    checkpoints: list[Path],
    *,
    device: torch.device,
    horizons: tuple[float, ...],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    seed_rows: list[list[dict[str, Any]]] = []
    for checkpoint in checkpoints:
        predictions, _ = _predict_learned(checkpoint, samples, device)
        current = []
        for sample in samples:
            image_id = str(sample["image_id"])
            fixations, observers = _load_fixations(sample)
            for index, horizon in enumerate(horizons):
                current.append(
                    _prediction_row(
                        np.asarray(predictions[image_id][index], dtype=np.float64),
                        sample,
                        horizon_index=index,
                        horizon=horizon,
                        budgets=budgets,
                        fixations=fixations,
                        observers=observers,
                    )
                )
        seed_rows.append(current)
    return _average_rows(seed_rows, (*METRICS, *(f"coverage_{budget:g}" for budget in budgets)))


def _average_rows(rows_by_run: list[list[dict[str, Any]]], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_run:
        for row in rows:
            grouped[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            **{key: float(np.mean([float(row[key]) for row in current])) for key in value_keys},
        }
        for (content, horizon), current in sorted(grouped.items())
    ]


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    horizons: tuple[float, ...],
    primary: set[float],
    budgets: tuple[float, ...],
) -> dict[str, Any]:
    per_horizon = {}
    keys = (*METRICS, *(f"coverage_{budget:g}" for budget in budgets))
    for horizon in horizons:
        current = [row for row in rows if np.isclose(float(row["horizon_s"]), horizon)]
        per_horizon[f"{horizon:g}"] = {
            key: float(np.mean([float(row[key]) for row in current])) for key in keys
        }
    early = [row for row in rows if float(row["horizon_s"]) in primary]
    return {
        "content_count": len({str(row["content_id"]) for row in rows}),
        "early_macro": {key: float(np.mean([float(row[key]) for row in early])) for key in keys},
        "per_horizon": per_horizon,
    }


def _fit_alphas(
    samples: list[dict[str, Any]],
    semantic: dict[str, np.ndarray],
    *,
    horizons: tuple[float, ...],
    budgets: tuple[float, ...],
    selection_budgets: tuple[float, ...],
    alpha_grid: tuple[float, ...],
    kld_guard: float,
) -> tuple[dict[float, float], dict[str, list[dict[str, float]]]]:
    chosen: dict[float, float] = {}
    traces: dict[str, list[dict[str, float]]] = {}
    for horizon_index, horizon in enumerate(horizons):
        current_samples = []
        template_klds = []
        for sample in samples:
            template = np.asarray(sample["template"].numpy()[horizon_index], dtype=np.float64)
            target = np.asarray(sample["target"].numpy()[horizon_index], dtype=np.float64)
            area = np.asarray(sample["area_weights"].numpy()[0], dtype=np.float64)
            template_klds.append(kld(template, target, area))
            current_samples.append((sample, template, target, area, *_load_fixations(sample)))
        template_kld = float(np.mean(template_klds))
        candidates = []
        for alpha in alpha_grid:
            klds, coverage = [], {budget: [] for budget in budgets}
            for sample, template, target, area, fixations, observers in current_samples:
                prediction = geometric_semantic_fusion(
                    template,
                    semantic[str(sample["image_id"])],
                    area,
                    alpha=float(alpha),
                )
                klds.append(kld(prediction, target, area))
                for budget in budgets:
                    selected = rank_tiles(
                        prediction, area, tile_rows=8, tile_cols=16, budget_percent=budget
                    )
                    coverage[budget].append(
                        gaze_mass_coverage(
                            fixations,
                            observer_ids=observers,
                            horizon_s=float(horizon),
                            selected_tile_ids=selected,
                            tile_rows=8,
                            tile_cols=16,
                        )
                    )
            mean_kld = float(np.mean(klds))
            score = float(np.mean([np.mean(coverage[budget]) for budget in selection_budgets]))
            candidates.append(
                {
                    "alpha": float(alpha),
                    "kld_sph": mean_kld,
                    "coverage_selection_mean": score,
                    "passes_kld_guard": float(mean_kld <= template_kld + float(kld_guard)),
                    **{f"coverage_{budget:g}": float(np.mean(values)) for budget, values in coverage.items()},
                }
            )
        feasible = [row for row in candidates if bool(row["passes_kld_guard"])]
        if not feasible:
            feasible = [row for row in candidates if np.isclose(row["alpha"], 0.0)]
        selected = max(feasible, key=lambda row: (row["coverage_selection_mean"], -row["kld_sph"], -row["alpha"]))
        chosen[float(horizon)] = float(selected["alpha"])
        traces[f"{horizon:g}"] = candidates
    return chosen, traces


def _tempsal_rows(path: Path, budgets: tuple[float, ...]) -> list[dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, float], dict[str, Any]] = {}
    for row in result["score_rows"]:
        key = (str(row["content_id"]), float(row["horizon_s"]))
        grouped[key] = {
            "content_id": key[0],
            "horizon_s": key[1],
            **{metric: float(row[metric]) for metric in METRICS},
        }
    for budget in budgets:
        for row in result["coverage_rows"][str(float(budget))]:
            key = (str(row["content_id"]), float(row["horizon_s"]))
            grouped[key][f"coverage_{budget:g}"] = float(row["coverage"])
    return [grouped[key] for key in sorted(grouped)]


def _paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    horizon: float,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    left_map = {
        str(row["content_id"]): float(row[metric])
        for row in left
        if np.isclose(float(row["horizon_s"]), horizon)
    }
    right_map = {
        str(row["content_id"]): float(row[metric])
        for row in right
        if np.isclose(float(row["horizon_s"]), horizon)
    }
    contents = sorted(set(left_map) & set(right_map))
    differences = np.asarray([left_map[key] - right_map[key] for key in contents], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(differences), size=(int(iterations), len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "mean": float(np.mean(differences)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "content_count": len(contents),
    }


def _write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S0 Semantic Re-ranking Headroom Diagnostic",
        "",
        f"Decision: **{result['decision']}**. Alpha values are selected on Salient360 train only; all displayed scores are on Salient360 val. Gate/test target loads are zero.",
        "",
        "## Selected semantic mixing coefficients",
        "",
        "| Source | 0.5 s | 1 s | 3 s | 5 s |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in ("gt_final_oracle", "deepgaze_raw"):
        values = result["selected_alphas"][source]
        lines.append(
            f"| {source} | {values['0.5']:.2f} | {values['1']:.2f} | {values['3']:.2f} | {values['5']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Validation comparison at 3 s",
            "",
            "| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("template", "spherehorizon", "tempsal", "gt_final_oracle", "wrong_image_gt_final", "deepgaze_raw"):
        row = result["methods"][method]["per_horizon"]["3"]
        lines.append(
            f"| {method} | {row['kld_sph']:.4f} | {row['sim_sph']:.4f} | {row['cc_sph']:.4f} | "
            f"{row['coverage_5']:.4f} | {row['coverage_10']:.4f} | {row['coverage_20']:.4f} |"
        )
    lines.extend(["", "## Paired 3 s coverage evidence", ""])
    for comparison, metrics in result["comparisons"].items():
        for metric, values in metrics.items():
            lines.append(
                f"- {comparison}, {metric}: {values['mean']:+.4f} "
                f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}]"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The GT-final row is an oracle diagnostic, not an inference result. It asks whether image-specific semantic evidence can re-rank the start/time template.",
            "- The wrong-image row uses the same alpha as the same-image oracle, so the gap isolates content correspondence rather than extra tuning.",
            "- DeepGaze raw is a frozen practical semantic source. Its role is diagnostic; it is not adopted as the proposed model.",
            "- Passing S0 authorizes a learned semantic re-ranking pilot, not a gate/test claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate no-training semantic re-ranking headroom.")
    parser.add_argument("--config", default="configs/semantic_reranking_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = ROOT / config["data"]["root"]
    train_split = str(config["data"]["selection_split"])
    val_split = str(config["data"]["evaluation_split"])
    horizons = tuple(float(value) for value in config["data"]["horizons_s"])
    primary = {float(value) for value in config["data"]["primary_horizons_s"]}
    budgets = tuple(float(value) for value in config["coverage"]["budgets_percent"])
    selection_budgets = tuple(float(value) for value in config["coverage"]["selection_budgets_percent"])
    alpha_grid = tuple(float(value) for value in config["fusion"]["alpha_grid"])
    deepgaze_root = ROOT / config["semantic_sources"]["deepgaze_raw"]["root"]
    train_samples = _samples(data_root, train_split)
    val_samples = _samples(data_root, val_split)

    selected_alphas: dict[str, dict[float, float]] = {}
    alpha_traces = {}
    for source in ("gt_final_oracle", "deepgaze_raw"):
        semantic = _semantic_maps(
            train_samples,
            source=source,
            split=train_split,
            deepgaze_root=deepgaze_root,
        )
        selected, trace = _fit_alphas(
            train_samples,
            semantic,
            horizons=horizons,
            budgets=budgets,
            selection_budgets=selection_budgets,
            alpha_grid=alpha_grid,
            kld_guard=float(config["fusion"]["kld_guard_over_template"]),
        )
        selected_alphas[source] = selected
        alpha_traces[source] = trace

    all_zero = {horizon: 0.0 for horizon in horizons}
    val_semantics = {
        source: _semantic_maps(
            val_samples,
            source=source,
            split=val_split,
            deepgaze_root=deepgaze_root,
        )
        for source in ("gt_final_oracle", "wrong_image_gt_final", "deepgaze_raw")
    }
    method_rows = {
        "template": _evaluate_fusion(
            val_samples,
            val_semantics["gt_final_oracle"],
            all_zero,
            horizons=horizons,
            budgets=budgets,
        ),
        "gt_final_oracle": _evaluate_fusion(
            val_samples,
            val_semantics["gt_final_oracle"],
            selected_alphas["gt_final_oracle"],
            horizons=horizons,
            budgets=budgets,
        ),
        "wrong_image_gt_final": _evaluate_fusion(
            val_samples,
            val_semantics["wrong_image_gt_final"],
            selected_alphas["gt_final_oracle"],
            horizons=horizons,
            budgets=budgets,
        ),
        "deepgaze_raw": _evaluate_fusion(
            val_samples,
            val_semantics["deepgaze_raw"],
            selected_alphas["deepgaze_raw"],
            horizons=horizons,
            budgets=budgets,
        ),
    }
    method_rows["spherehorizon"] = _evaluate_checkpoints(
        val_samples,
        [ROOT / value for value in config["references"]["spherehorizon_checkpoints"]],
        device=torch.device(str(config["references"]["device"])),
        horizons=horizons,
        budgets=budgets,
    )
    method_rows["tempsal"] = _tempsal_rows(ROOT / config["references"]["tempsal_report"], budgets)

    methods = {
        name: _aggregate(rows, horizons=horizons, primary=primary, budgets=budgets)
        for name, rows in method_rows.items()
    }
    decision_horizon = float(config["decision"]["horizon_s"])
    iterations = int(config["output"]["bootstrap_iterations"])
    seed = int(config["output"]["bootstrap_seed"])
    comparisons = {}
    for label, right in (
        ("oracle_minus_template", "template"),
        ("oracle_minus_wrong_image", "wrong_image_gt_final"),
        ("oracle_minus_tempsal", "tempsal"),
    ):
        comparisons[label] = {
            f"coverage_{budget:g}": _paired_bootstrap(
                method_rows["gt_final_oracle"],
                method_rows[right],
                horizon=decision_horizon,
                metric=f"coverage_{budget:g}",
                iterations=iterations,
                seed=seed + index,
            )
            for index, budget in enumerate(selection_budgets)
        }
    oracle_template = comparisons["oracle_minus_template"]
    oracle_wrong = comparisons["oracle_minus_wrong_image"]
    gain5 = oracle_template["coverage_5"]["mean"]
    gain10 = oracle_template["coverage_10"]["mean"]
    wrong_positive = all(values["ci95_low"] > 0.0 for values in oracle_wrong.values())
    passed = (
        gain5 >= float(config["decision"]["min_oracle_gain_over_template_at_5_percent"])
        and gain10 >= float(config["decision"]["min_oracle_gain_over_template_at_10_percent"])
        and (
            wrong_positive
            if bool(config["decision"]["require_oracle_better_than_wrong_image_ci95"])
            else True
        )
    )
    output = ensure_dir(ROOT / config["output"]["root"])
    result = {
        "protocol": config["protocol"],
        "decision": "PASS_AUTHORIZE_S1" if passed else "FAIL_STOP_METHOD_EXPANSION",
        "selected_alphas": {
            source: {f"{horizon:g}": value for horizon, value in values.items()}
            for source, values in selected_alphas.items()
        },
        "alpha_selection_traces": alpha_traces,
        "methods": methods,
        "comparisons": comparisons,
        "decision_checks": {
            "oracle_gain_cov5": gain5,
            "oracle_gain_cov10": gain10,
            "oracle_better_than_wrong_image_ci95": wrong_positive,
            "passed": passed,
        },
        "data_access": {
            "selection_split": train_split,
            "evaluation_split": val_split,
            "train_target_loads": len(train_samples),
            "val_target_loads": len(val_samples),
            "gate_target_loads": 0,
            "sealed_test_target_loads": int(config["output"]["sealed_test_target_loads"]),
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["method", "content_id", "horizon_s", *METRICS, *(f"coverage_{budget:g}" for budget in budgets)]
    with (output / "per_content_horizon.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, rows in method_rows.items():
            for row in rows:
                writer.writerow({"method": method, **{key: row[key] for key in fields if key != "method"}})
    _write_summary(result, output / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
