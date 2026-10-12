from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from sphererate.data.early_horizon import (
    build_cumulative_density,
    horizon_key,
    raw_dwell_tile_distribution,
)
from sphererate.data.maps import MapGeometry
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.temporal_factorization import deterministic_observer_split


IMPROVEMENT_METRICS = ("kld_improvement", "sim_improvement", "cc_improvement")
ROW_METADATA = {"content_id", "horizon_s", "observer_count", "observer_splits"}


def load_non_test_manifest(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    outer_test_metadata_rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) == "test":
                outer_test_metadata_rows += 1
                continue
            rows.append(row)
    if any(str(row.get("split")) == "test" for row in rows):
        raise RuntimeError("Outer-test row entered EH0 selection")
    return rows, {
        "manifest_rows": len(rows) + outer_test_metadata_rows,
        "selected_non_test_rows": len(rows),
        "outer_test_metadata_rows": outer_test_metadata_rows,
        "outer_test_target_rows_loaded": 0,
    }


def _metrics(prediction: np.ndarray, target: np.ndarray, area: np.ndarray) -> dict[str, float]:
    return {
        "kld": kld(prediction, target, area),
        "sim": sim(prediction, target, area),
        "cc": cc(prediction, target, area),
    }


def _spearman_brown(split_half_cc: float) -> float:
    """Approximate full-cohort map reliability from split-half correlation."""

    value = float(np.clip(split_half_cc, -0.999999, 1.0))
    return float(np.clip((2.0 * value) / (1.0 + value), -1.0, 1.0))


def _tile_count(tile_rows: int, tile_cols: int, budget_percent: float) -> int:
    total = int(tile_rows) * int(tile_cols)
    if total <= 0:
        raise ValueError("tile grid must be positive")
    budget = float(budget_percent)
    if not 0.0 < budget <= 100.0:
        raise ValueError("tile budget must be in (0, 100]")
    return min(total, max(1, int(np.ceil(total * budget / 100.0))))


def _top_tile_ids(mass: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    tile_ids = np.arange(values.size, dtype=np.int64)
    return np.lexsort((tile_ids, -values))[: int(count)].astype(np.int64)


def _density_tile_mass(
    density: np.ndarray,
    area: np.ndarray,
    *,
    tile_rows: int,
    tile_cols: int,
) -> np.ndarray:
    height, width = np.asarray(density).shape
    if height % int(tile_rows) or width % int(tile_cols):
        raise ValueError("map dimensions must be divisible by tile grid")
    tile_h = height // int(tile_rows)
    tile_w = width // int(tile_cols)
    return (
        (np.asarray(density, dtype=np.float64) * np.asarray(area, dtype=np.float64))
        .reshape(int(tile_rows), tile_h, int(tile_cols), tile_w)
        .sum(axis=(1, 3))
    )


def evaluate_content_reliability(
    *,
    content_id: str,
    fixations: np.ndarray,
    horizons_s: Sequence[float],
    height: int,
    width: int,
    sigma_px: float | None,
    observer_splits: int,
    seed: int,
    renderer: str = "legacy_erp",
    sigma_deg: float | None = None,
    templates: dict[float, np.ndarray] | None = None,
    budgets_percent: Sequence[float] = (),
    tile_rows: int = 8,
    tile_cols: int = 16,
) -> list[dict[str, float | str | int]]:
    observer_ids = sorted(np.unique(np.asarray(fixations)[:, 3]).astype(int).tolist())
    if len(observer_ids) < 4:
        raise ValueError(f"{content_id} needs at least four physical observers, found {len(observer_ids)}")
    if int(observer_splits) <= 0:
        raise ValueError("observer_splits must be positive")
    area = MapGeometry(int(height), int(width)).area_weights
    collected: dict[float, list[dict[str, float]]] = defaultdict(list)
    template_maps = {
        float(key): np.asarray(value, dtype=np.float32)
        for key, value in (templates or {}).items()
    }
    missing_templates = {float(value) for value in horizons_s} - set(template_maps)
    if templates is not None and missing_templates:
        raise KeyError(f"Missing templates for horizons {sorted(missing_templates)}")
    template_tile_ids: dict[tuple[float, float], np.ndarray] = {}
    for horizon, template in template_maps.items():
        if template.shape != (int(height), int(width)):
            raise ValueError(
                f"Template at {horizon:g}s has shape {template.shape}, expected {(int(height), int(width))}"
            )
        tile_mass = _density_tile_mass(
            template,
            area,
            tile_rows=int(tile_rows),
            tile_cols=int(tile_cols),
        )
        for budget in budgets_percent:
            count = _tile_count(tile_rows, tile_cols, float(budget))
            template_tile_ids[(horizon, float(budget))] = _top_tile_ids(tile_mass, count)

    for split_index in range(int(observer_splits)):
        left, right = deterministic_observer_split(
            observer_ids,
            key=f"{content_id}:observer-split:{split_index}",
            seed=int(seed),
        )
        for source_ids, target_ids in ((left, right), (right, left)):
            static_prediction = build_cumulative_density(
                fixations,
                observer_ids=source_ids,
                horizon_s=None,
                height=height,
                width=width,
                sigma_px=sigma_px,
                renderer=renderer,
                sigma_deg=sigma_deg,
            )
            for horizon in horizons_s:
                same_prediction = build_cumulative_density(
                    fixations,
                    observer_ids=source_ids,
                    horizon_s=float(horizon),
                    height=height,
                    width=width,
                    sigma_px=sigma_px,
                    renderer=renderer,
                    sigma_deg=sigma_deg,
                )
                target = build_cumulative_density(
                    fixations,
                    observer_ids=target_ids,
                    horizon_s=float(horizon),
                    height=height,
                    width=width,
                    sigma_px=sigma_px,
                    renderer=renderer,
                    sigma_deg=sigma_deg,
                )
                same = _metrics(same_prediction, target, area)
                static = _metrics(static_prediction, target, area)
                values = {
                    "same_kld": same["kld"],
                    "same_sim": same["sim"],
                    "same_cc": same["cc"],
                    "static_kld": static["kld"],
                    "static_sim": static["sim"],
                    "static_cc": static["cc"],
                    "kld_improvement": static["kld"] - same["kld"],
                    "sim_improvement": same["sim"] - static["sim"],
                    "cc_improvement": same["cc"] - static["cc"],
                }
                if template_maps:
                    template_scores = _metrics(template_maps[float(horizon)], target, area)
                    values.update(
                        {
                            "template_kld": template_scores["kld"],
                            "template_sim": template_scores["sim"],
                            "template_cc": template_scores["cc"],
                            "same_vs_template_kld_improvement": template_scores["kld"] - same["kld"],
                            "same_vs_template_sim_improvement": same["sim"] - template_scores["sim"],
                            "same_vs_template_cc_improvement": same["cc"] - template_scores["cc"],
                        }
                    )
                if budgets_percent:
                    source_mass = raw_dwell_tile_distribution(
                        fixations,
                        observer_ids=source_ids,
                        horizon_s=float(horizon),
                        tile_rows=int(tile_rows),
                        tile_cols=int(tile_cols),
                    ).reshape(-1)
                    target_mass = raw_dwell_tile_distribution(
                        fixations,
                        observer_ids=target_ids,
                        horizon_s=float(horizon),
                        tile_rows=int(tile_rows),
                        tile_cols=int(tile_cols),
                    ).reshape(-1)
                    for budget in budgets_percent:
                        budget_value = float(budget)
                        budget_key = f"{budget_value:g}"
                        count = _tile_count(tile_rows, tile_cols, budget_value)
                        source_top = _top_tile_ids(source_mass, count)
                        target_top = _top_tile_ids(target_mass, count)
                        intersection = len(set(source_top.tolist()) & set(target_top.tolist()))
                        union = len(set(source_top.tolist()) | set(target_top.tolist()))
                        cross_coverage = float(target_mass[source_top].sum())
                        oracle_coverage = float(target_mass[target_top].sum())
                        values[f"tile_overlap_{budget_key}"] = float(intersection / count)
                        values[f"tile_jaccard_{budget_key}"] = float(intersection / union)
                        values[f"tile_cross_coverage_{budget_key}"] = cross_coverage
                        values[f"tile_oracle_coverage_{budget_key}"] = oracle_coverage
                        values[f"tile_normalized_coverage_{budget_key}"] = float(
                            cross_coverage / max(oracle_coverage, 1e-12)
                        )
                        if template_maps:
                            template_top = template_tile_ids[(float(horizon), budget_value)]
                            template_coverage = float(target_mass[template_top].sum())
                            values[f"tile_template_coverage_{budget_key}"] = template_coverage
                            values[f"tile_template_normalized_coverage_{budget_key}"] = float(
                                template_coverage / max(oracle_coverage, 1e-12)
                            )
                            values[f"tile_same_vs_template_coverage_{budget_key}"] = (
                                cross_coverage - template_coverage
                            )
                collected[float(horizon)].append(values)

    rows: list[dict[str, float | str | int]] = []
    for horizon in horizons_s:
        values = collected[float(horizon)]
        row: dict[str, float | str | int] = {
            "content_id": str(content_id),
            "horizon_s": float(horizon),
            "observer_count": len(observer_ids),
            "observer_splits": int(observer_splits),
        }
        for name in values[0]:
            row[name] = float(np.mean([value[name] for value in values]))
        row["full_cohort_cc_estimate"] = _spearman_brown(float(row["same_cc"]))
        rows.append(row)
    return rows


def bootstrap_mean_ci(
    values_by_content: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    if not values_by_content:
        raise ValueError("No content values supplied")
    values = np.asarray([values_by_content[key] for key in sorted(values_by_content)], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(iterations), len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "num_contents": int(len(values)),
    }


def aggregate_reliability(
    rows: Iterable[dict[str, Any]],
    *,
    primary_horizons_s: Sequence[float] = (0.5, 1.0, 3.0),
    iterations: int = 10000,
    seed: int = 20260711,
) -> dict[str, Any]:
    records = list(rows)
    by_horizon: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_horizon[float(row["horizon_s"])].append(row)
    per_horizon: dict[str, Any] = {}
    metric_names = sorted(
        key
        for key, value in records[0].items()
        if key not in ROW_METADATA and isinstance(value, (int, float, np.integer, np.floating))
    )
    for horizon, current in sorted(by_horizon.items()):
        per_horizon[horizon_key(horizon)] = {
            metric: bootstrap_mean_ci(
                {str(row["content_id"]): float(row[metric]) for row in current},
                iterations=iterations,
                seed=seed,
            )
            for metric in metric_names
        }

    primary = {float(value) for value in primary_horizons_s}
    content_metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if float(row["horizon_s"]) not in primary:
            continue
        for metric in metric_names:
            content_metric_values[str(row["content_id"])][metric].append(float(row[metric]))
    early_macro = {
        metric: bootstrap_mean_ci(
            {
                content: float(np.mean(metrics[metric]))
                for content, metrics in content_metric_values.items()
                if len(metrics[metric]) == len(primary)
            },
            iterations=iterations,
            seed=seed,
        )
        for metric in metric_names
    }
    return {"early_macro": early_macro, "per_horizon": per_horizon}


def decide_eh0(
    aggregates: dict[str, Any],
    *,
    canonical_start_verified: bool,
    outer_test_used: bool,
) -> dict[str, Any]:
    macro = aggregates["early_macro"]
    per_horizon = aggregates["per_horizon"]
    individually_positive = 0
    for horizon in (0.5, 1.0, 3.0):
        scores = per_horizon[horizon_key(horizon)]
        if scores["kld_improvement"]["ci95_low"] > 0.0 and scores["sim_improvement"]["ci95_low"] > 0.0:
            individually_positive += 1
    checks = {
        "early_macro_kld_positive": bool(macro["kld_improvement"]["ci95_low"] > 0.0),
        "early_macro_sim_positive": bool(macro["sim_improvement"]["ci95_low"] > 0.0),
        "at_least_two_early_horizons_positive": bool(individually_positive >= 2),
        "five_second_cc_guard": bool(per_horizon["5"]["cc_improvement"]["ci95_low"] >= -0.01),
        "canonical_start_verified": bool(canonical_start_verified),
        "outer_test_not_used": not bool(outer_test_used),
    }
    return {"decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
