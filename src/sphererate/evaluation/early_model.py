from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from sphererate.data.early_horizon import horizon_key
from sphererate.evaluation.early_reliability import bootstrap_mean_ci


def compare_method_rows(
    candidate_rows: Iterable[dict[str, Any]],
    control_rows: Iterable[dict[str, Any]],
    *,
    metrics: Sequence[str] = ("kld_sph", "sim_sph", "cc_sph"),
    primary_horizons_s: Sequence[float] = (0.5, 1.0, 3.0),
    iterations: int = 10000,
    seed: int = 20260711,
) -> dict[str, Any]:
    candidate = {(str(row["content_id"]), float(row["horizon_s"])): row for row in candidate_rows}
    control = {(str(row["content_id"]), float(row["horizon_s"])): row for row in control_rows}
    if set(candidate) != set(control):
        raise ValueError("Candidate/control rows are not paired")
    oriented: dict[tuple[str, float], dict[str, float]] = {}
    for key in candidate:
        oriented[key] = {}
        for metric in metrics:
            if metric.startswith("kld"):
                value = float(control[key][metric]) - float(candidate[key][metric])
            else:
                value = float(candidate[key][metric]) - float(control[key][metric])
            oriented[key][metric] = value
    per_horizon: dict[str, Any] = {}
    horizons = sorted({key[1] for key in oriented})
    for horizon in horizons:
        per_horizon[horizon_key(horizon)] = {
            metric: bootstrap_mean_ci(
                {content: values[metric] for (content, current), values in oriented.items() if current == horizon},
                iterations=iterations,
                seed=seed,
            )
            for metric in metrics
        }
    primary = {float(value) for value in primary_horizons_s}
    by_content: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (content, horizon), values in oriented.items():
        if horizon in primary:
            for metric in metrics:
                by_content[content][metric].append(values[metric])
    early_macro = {
        metric: bootstrap_mean_ci(
            {
                content: float(np.mean(values[metric]))
                for content, values in by_content.items()
                if len(values[metric]) == len(primary)
            },
            iterations=iterations,
            seed=seed,
        )
        for metric in metrics
    }
    return {"early_macro": early_macro, "per_horizon": per_horizon}


def compare_coverage_rows(
    candidate_rows: Iterable[dict[str, Any]],
    control_rows: Iterable[dict[str, Any]],
    *,
    primary_horizons_s: Sequence[float] = (0.5, 1.0, 3.0),
    iterations: int = 10000,
    seed: int = 20260711,
) -> dict[str, Any]:
    candidate = {(str(row["content_id"]), float(row["horizon_s"])): float(row["coverage"]) for row in candidate_rows}
    control = {(str(row["content_id"]), float(row["horizon_s"])): float(row["coverage"]) for row in control_rows}
    if set(candidate) != set(control):
        raise ValueError("Coverage rows are not paired")
    primary = {float(value) for value in primary_horizons_s}
    by_content: dict[str, list[float]] = defaultdict(list)
    for (content, horizon), value in candidate.items():
        if horizon in primary:
            by_content[content].append(value - control[(content, horizon)])
    return bootstrap_mean_ci(
        {content: float(np.mean(values)) for content, values in by_content.items() if len(values) == len(primary)},
        iterations=iterations,
        seed=seed,
    )


def decide_eh3(
    comparisons: dict[str, Any],
    *,
    coverage_comparisons: dict[str, dict[str, Any]],
    parameter_match: bool,
    outer_test_used: bool,
    cc_floor: float = -0.01,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for control in ("static_final_repeat", "early_template"):
        macro = comparisons[control]["early_macro"]
        checks[f"map_vs_{control}_kld"] = bool(macro["kld_sph"]["ci95_low"] > 0.0)
        checks[f"map_vs_{control}_sim"] = bool(macro["sim_sph"]["ci95_low"] > 0.0)
    for control in ("candidate_shuffled_label", "wrong_time", "wrong_start", "wrong_image"):
        macro = comparisons[control]["early_macro"]
        kld = macro["kld_sph"]
        sim = macro["sim_sph"]
        checks[f"attribution_{control}"] = bool(
            (kld["ci95_low"] > 0.0 and sim["mean"] > 0.0)
            or (sim["ci95_low"] > 0.0 and kld["mean"] > 0.0)
        )
    for control, result in coverage_comparisons.items():
        checks[f"coverage_vs_{control}"] = bool(result["ci95_low"] > 0.0)
    for control, comparison in comparisons.items():
        checks[f"cc_guard_{control}"] = bool(
            comparison["early_macro"]["cc_sph"]["ci95_low"] >= float(cc_floor)
        )
    checks["matched_parameter_counts"] = bool(parameter_match)
    checks["outer_test_not_used"] = not bool(outer_test_used)
    return {"decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
