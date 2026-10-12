from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from sphererate.evaluation.late_saliency import fuse_log_density
from sphererate.evaluation.metrics import kld


def validate_r1_splits(selection_split: str, evaluation_split: str) -> None:
    if str(selection_split) != "train":
        raise ValueError(f"R1 selection split must be train, got {selection_split!r}")
    if str(evaluation_split) == "test":
        raise ValueError("R1 refuses the test split")
    if str(evaluation_split) != "val":
        raise ValueError(f"R1 evaluation split must be val, got {evaluation_split!r}")


def wrong_start_longitude(longitude_deg: float) -> float:
    return float(((float(longitude_deg) + 360.0) % 360.0) - 180.0)


def content_oracle_metadata() -> dict[str, Any]:
    return {
        "name": "content_oracle",
        "source": "Salient360 population saliency map from independent observers",
        "deployable": False,
        "role": "R1 signal feasibility only",
    }


def select_fusion_alpha(
    train_samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    alpha_grid: Iterable[float],
    area_weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    samples = list(train_samples)
    if not samples:
        raise ValueError("R1 alpha selection requires train samples")
    grid = [float(value) for value in alpha_grid]
    if not grid:
        raise ValueError("R1 alpha grid is empty")
    scores: dict[str, float] = {}
    for alpha in grid:
        values = []
        for target, start_map, content_map in samples:
            prediction = fuse_log_density(start_map, content_map, alpha, area_weights)
            values.append(kld(prediction, target, area_weights))
        scores[f"{alpha:.3f}"] = float(np.mean(values))
    selected = min(grid, key=lambda value: (scores[f"{value:.3f}"], value))
    return float(selected), scores


def decide_r1(
    *,
    selected_alpha: float,
    comparisons: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    def positive(comparison: str, metric: str) -> bool:
        return float(comparisons[comparison][metric]["ci95_low"]) > 0.0

    checks = {
        "fusion_vs_start_kld": positive("fusion_vs_start", "kld_sph"),
        "fusion_vs_start_sim": positive("fusion_vs_start", "sim_sph"),
        "fusion_vs_wrong_image_kld": positive("fusion_vs_wrong_image", "kld_sph"),
        "fusion_vs_wrong_image_sim": positive("fusion_vs_wrong_image", "sim_sph"),
        "start_vs_wrong180_kld": positive("start_vs_wrong180", "kld_sph"),
        "start_vs_wrong180_sim": positive("start_vs_wrong180", "sim_sph"),
        "fusion_vs_start_cc_guard": float(comparisons["fusion_vs_start"]["cc_sph"]["ci95_low"]) >= -0.01,
        "interior_alpha": 0.0 < float(selected_alpha) < 1.0,
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "selected_alpha": float(selected_alpha),
        "checks": checks,
    }
