from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from sphererate.data.early_horizon import (
    build_cumulative_density,
    horizon_key,
    raw_dwell_tile_distribution,
)
from sphererate.data.maps import MapGeometry
from sphererate.evaluation.early_reliability import bootstrap_mean_ci
from sphererate.evaluation.temporal_factorization import deterministic_observer_split


ORACLE_IMPROVEMENTS = ("same_vs_static_improvement", "same_vs_template_improvement")


def budget_tile_count(tile_rows: int, tile_cols: int, budget_percent: float) -> int:
    total = int(tile_rows) * int(tile_cols)
    if total <= 0:
        raise ValueError("tile grid must be positive")
    percent = float(budget_percent)
    if not 0.0 < percent <= 100.0:
        raise ValueError("budget_percent must be in (0, 100]")
    return min(total, max(1, int(np.ceil(total * percent / 100.0))))


def tile_probability_mass(
    density: np.ndarray,
    area_weights: np.ndarray,
    *,
    tile_rows: int,
    tile_cols: int,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    area = np.asarray(area_weights, dtype=np.float64)
    if density.ndim != 2 or density.shape != area.shape:
        raise ValueError("density and area_weights must be matching 2D arrays")
    height, width = density.shape
    if height % int(tile_rows) or width % int(tile_cols):
        raise ValueError("map dimensions must be divisible by tile grid")
    tile_h = height // int(tile_rows)
    tile_w = width // int(tile_cols)
    mass = (density * area).reshape(int(tile_rows), tile_h, int(tile_cols), tile_w).sum(axis=(1, 3))
    return mass.astype(np.float64)


def rank_tiles(
    density: np.ndarray,
    area_weights: np.ndarray,
    *,
    tile_rows: int,
    tile_cols: int,
    budget_percent: float,
) -> np.ndarray:
    scores = tile_probability_mass(
        density,
        area_weights,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    ).ravel()
    tile_ids = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((tile_ids, -scores))
    return order[: budget_tile_count(tile_rows, tile_cols, budget_percent)].astype(np.int64)


def gaze_mass_coverage(
    fixations: np.ndarray,
    *,
    observer_ids: Iterable[int],
    horizon_s: float,
    selected_tile_ids: Iterable[int],
    tile_rows: int,
    tile_cols: int,
) -> float:
    array = np.asarray(fixations, dtype=np.float32)
    selected_observers = np.asarray(sorted({int(value) for value in observer_ids}), dtype=np.int64)
    distribution = raw_dwell_tile_distribution(
        array,
        observer_ids=selected_observers,
        horizon_s=float(horizon_s),
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
    ).ravel()
    selected = np.asarray(sorted({int(value) for value in selected_tile_ids}), dtype=np.int64)
    selected = selected[(selected >= 0) & (selected < distribution.size)]
    return float(distribution[selected].sum())


def evaluate_content_tile_oracle(
    *,
    content_id: str,
    fixations: np.ndarray,
    templates: dict[float, np.ndarray],
    priors: dict[str, np.ndarray],
    horizons_s: Sequence[float],
    budgets_percent: Sequence[float],
    height: int,
    width: int,
    sigma_px: float,
    tile_rows: int,
    tile_cols: int,
    observer_splits: int,
    seed: int,
) -> list[dict[str, Any]]:
    observer_ids = sorted(np.unique(np.asarray(fixations)[:, 3]).astype(int).tolist())
    if len(observer_ids) < 4:
        raise ValueError(f"{content_id} needs at least four observers")
    area = MapGeometry(int(height), int(width)).area_weights
    rankings: dict[tuple[str, float, float], np.ndarray] = {}
    for horizon in horizons_s:
        if float(horizon) not in templates:
            raise KeyError(f"Missing template for horizon {horizon}")
        for budget in budgets_percent:
            rankings[("template", float(horizon), float(budget))] = rank_tiles(
                templates[float(horizon)], area, tile_rows=tile_rows, tile_cols=tile_cols, budget_percent=budget
            )
            for name, prior in priors.items():
                rankings[(name, float(horizon), float(budget))] = rank_tiles(
                    prior, area, tile_rows=tile_rows, tile_cols=tile_cols, budget_percent=budget
                )

    collected: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for split_index in range(int(observer_splits)):
        left, right = deterministic_observer_split(
            observer_ids,
            key=f"{content_id}:tile-oracle:{split_index}",
            seed=int(seed),
        )
        for source_ids, target_ids in ((left, right), (right, left)):
            static_map = build_cumulative_density(
                fixations,
                observer_ids=source_ids,
                horizon_s=None,
                height=height,
                width=width,
                sigma_px=sigma_px,
            )
            for horizon in horizons_s:
                same_map = build_cumulative_density(
                    fixations,
                    observer_ids=source_ids,
                    horizon_s=float(horizon),
                    height=height,
                    width=width,
                    sigma_px=sigma_px,
                )
                for budget in budgets_percent:
                    same_tiles = rank_tiles(
                        same_map, area, tile_rows=tile_rows, tile_cols=tile_cols, budget_percent=budget
                    )
                    static_tiles = rank_tiles(
                        static_map, area, tile_rows=tile_rows, tile_cols=tile_cols, budget_percent=budget
                    )
                    same_coverage = gaze_mass_coverage(
                        fixations,
                        observer_ids=target_ids,
                        horizon_s=float(horizon),
                        selected_tile_ids=same_tiles,
                        tile_rows=tile_rows,
                        tile_cols=tile_cols,
                    )
                    static_coverage = gaze_mass_coverage(
                        fixations,
                        observer_ids=target_ids,
                        horizon_s=float(horizon),
                        selected_tile_ids=static_tiles,
                        tile_rows=tile_rows,
                        tile_cols=tile_cols,
                    )
                    template_coverage = gaze_mass_coverage(
                        fixations,
                        observer_ids=target_ids,
                        horizon_s=float(horizon),
                        selected_tile_ids=rankings[("template", float(horizon), float(budget))],
                        tile_rows=tile_rows,
                        tile_cols=tile_cols,
                    )
                    values = {
                        "same_coverage": same_coverage,
                        "static_coverage": static_coverage,
                        "template_coverage": template_coverage,
                        "same_vs_static_improvement": same_coverage - static_coverage,
                        "same_vs_template_improvement": same_coverage - template_coverage,
                    }
                    for name in priors:
                        values[f"{name}_coverage"] = gaze_mass_coverage(
                            fixations,
                            observer_ids=target_ids,
                            horizon_s=float(horizon),
                            selected_tile_ids=rankings[(name, float(horizon), float(budget))],
                            tile_rows=tile_rows,
                            tile_cols=tile_cols,
                        )
                    collected[(float(horizon), float(budget))].append(values)

    rows: list[dict[str, Any]] = []
    for horizon in horizons_s:
        for budget in budgets_percent:
            values = collected[(float(horizon), float(budget))]
            row: dict[str, Any] = {
                "content_id": str(content_id),
                "horizon_s": float(horizon),
                "budget_percent": float(budget),
                "selected_tile_count": budget_tile_count(tile_rows, tile_cols, budget),
                "observer_count": len(observer_ids),
                "observer_splits": int(observer_splits),
            }
            for name in values[0]:
                row[name] = float(np.mean([value[name] for value in values]))
            rows.append(row)
    return rows


def aggregate_tile_oracle(
    rows: Iterable[dict[str, Any]],
    *,
    primary_budget_percent: float,
    primary_horizons_s: Sequence[float] = (0.5, 1.0, 3.0),
    iterations: int = 10000,
    seed: int = 20260711,
) -> dict[str, Any]:
    records = list(rows)
    budgets = sorted({float(row["budget_percent"]) for row in records})
    primary_horizons = {float(value) for value in primary_horizons_s}
    per_budget: dict[str, Any] = {}
    for budget in budgets:
        current_budget = [row for row in records if float(row["budget_percent"]) == budget]
        per_horizon: dict[str, Any] = {}
        for horizon in sorted({float(row["horizon_s"]) for row in current_budget}):
            current = [row for row in current_budget if float(row["horizon_s"]) == horizon]
            per_horizon[horizon_key(horizon)] = {
                metric: bootstrap_mean_ci(
                    {str(row["content_id"]): float(row[metric]) for row in current},
                    iterations=iterations,
                    seed=seed,
                )
                for metric in ORACLE_IMPROVEMENTS
            }
        by_content: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in current_budget:
            if float(row["horizon_s"]) not in primary_horizons:
                continue
            for metric in ORACLE_IMPROVEMENTS:
                by_content[str(row["content_id"])][metric].append(float(row[metric]))
        early_macro = {
            metric: bootstrap_mean_ci(
                {
                    content: float(np.mean(values[metric]))
                    for content, values in by_content.items()
                    if len(values[metric]) == len(primary_horizons)
                },
                iterations=iterations,
                seed=seed,
            )
            for metric in ORACLE_IMPROVEMENTS
        }
        per_budget[f"{budget:g}"] = {"per_horizon": per_horizon, "early_macro": early_macro}
    key = f"{float(primary_budget_percent):g}"
    if key not in per_budget:
        raise KeyError(f"Primary budget {key}% missing")
    return {
        "primary_budget_percent": float(primary_budget_percent),
        "primary": per_budget[key],
        "per_budget": per_budget,
    }


def decide_eh1(aggregates: dict[str, Any], *, outer_test_used: bool) -> dict[str, Any]:
    primary = aggregates["primary"]
    macro = primary["early_macro"]
    positive_horizons = sum(
        primary["per_horizon"][horizon_key(horizon)]["same_vs_static_improvement"]["ci95_low"] > 0.0
        for horizon in (0.5, 1.0, 3.0)
    )
    checks = {
        "early_macro_vs_static_positive": bool(
            macro["same_vs_static_improvement"]["ci95_low"] > 0.0
        ),
        "early_macro_vs_template_positive": bool(
            macro["same_vs_template_improvement"]["ci95_low"] > 0.0
        ),
        "at_least_two_early_horizons_vs_static_positive": bool(positive_horizons >= 2),
        "outer_test_not_used": not bool(outer_test_used),
    }
    return {"decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
