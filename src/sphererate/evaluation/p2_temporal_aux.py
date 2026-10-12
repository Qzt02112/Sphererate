from __future__ import annotations

from typing import Any

from sphererate.evaluation.content_interaction import grouped_bootstrap_improvement


CONTROL_ROLES = ("final_only", "temporal_shuffled", "capacity_final_repeat")
PRIMARY_METRICS = ("kld_sph", "sim_sph")


def p2_gate_decision(
    scores: dict[str, dict[str, dict[str, float]]],
    groups_by_image: dict[str, str],
    *,
    iterations: int = 10000,
    seed: int = 20260711,
    cc_floor: float = -0.01,
) -> dict[str, Any]:
    if "temporal_correct" not in scores:
        raise KeyError("Missing temporal_correct scores")
    comparisons: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for control in CONTROL_ROLES:
        if control not in scores:
            raise KeyError(f"Missing {control} scores")
        name = f"temporal_correct_vs_{control}"
        comparisons[name] = {}
        for metric in (*PRIMARY_METRICS, "cc_sph"):
            result = grouped_bootstrap_improvement(
                scores["temporal_correct"],
                scores[control],
                groups_by_image,
                metric,
                iterations=iterations,
                seed=seed,
            )
            comparisons[name][metric] = result
            threshold = float(cc_floor) if metric == "cc_sph" else 0.0
            checks[f"{name}_{metric}"] = bool(result["ci95_low"] >= threshold if metric == "cc_sph" else result["ci95_low"] > threshold)
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "comparisons": comparisons,
        "checks": checks,
        "test_used": False,
    }

