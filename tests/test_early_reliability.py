from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sphererate.evaluation.early_reliability import (
    aggregate_reliability,
    bootstrap_mean_ci,
    decide_eh0,
    evaluate_content_reliability,
    load_non_test_manifest,
)


def _synthetic_fixations() -> np.ndarray:
    rows: list[list[float]] = []
    for observer in range(8):
        jitter = (observer - 3.5) * 0.001
        rows.extend(
            [
                [0.20 + jitter, 0.5, 0.1, float(observer), 0.6, 0.0],
                [0.25 + jitter, 0.5, 0.8, float(observer), 0.6, 0.0],
                [0.75 + jitter, 0.5, 4.0, float(observer), 1.0, 0.0],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def test_content_reliability_is_deterministic_and_orients_improvement_positive() -> None:
    kwargs = dict(
        content_id="salient360:P1",
        fixations=_synthetic_fixations(),
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        height=16,
        width=32,
        sigma_px=1.0,
        observer_splits=3,
        seed=11,
    )
    first = evaluate_content_reliability(**kwargs)
    second = evaluate_content_reliability(**kwargs)

    assert first == second
    assert len(first) == 4
    assert first[0]["kld_improvement"] > 0
    assert first[0]["sim_improvement"] > 0
    assert first[0]["cc_improvement"] > 0
    assert first[0]["full_cohort_cc_estimate"] >= first[0]["same_cc"]


def test_content_reliability_reports_template_and_tile_agreement() -> None:
    template = np.ones((16, 32), dtype=np.float32)
    area = np.cos(
        np.pi / 2.0
        - np.pi * ((np.arange(16, dtype=np.float32) + 0.5) / 16.0)
    )[:, None]
    template /= float((template * area).sum())
    rows = evaluate_content_reliability(
        content_id="salient360:P1",
        fixations=_synthetic_fixations(),
        horizons_s=(0.5, 1.0),
        height=16,
        width=32,
        sigma_px=1.0,
        observer_splits=2,
        seed=11,
        templates={0.5: template, 1.0: template},
        budgets_percent=(10.0,),
        tile_rows=4,
        tile_cols=8,
    )

    assert "same_vs_template_kld_improvement" in rows[0]
    assert 0.0 <= rows[0]["tile_overlap_10"] <= 1.0
    assert 0.0 <= rows[0]["tile_normalized_coverage_10"] <= 1.0 + 1e-6

    aggregate = aggregate_reliability(
        rows,
        primary_horizons_s=(0.5, 1.0),
        iterations=100,
        seed=3,
    )
    assert "same_cc" in aggregate["per_horizon"]["0.5"]
    assert "tile_overlap_10" in aggregate["early_macro"]


def test_bootstrap_resamples_contents_not_metric_rows() -> None:
    values = {"P1": 1.0, "P2": 2.0, "P3": 3.0}
    result = bootstrap_mean_ci(values, iterations=1000, seed=7)

    assert result["mean"] == pytest.approx(2.0)
    assert 1.0 <= result["ci95_low"] <= result["ci95_high"] <= 3.0
    assert result["num_contents"] == 3


def test_eh0_decision_applies_preregistered_checks() -> None:
    positive = {"mean": 0.1, "ci95_low": 0.01, "ci95_high": 0.2, "num_contents": 10}
    neutral = {"mean": 0.0, "ci95_low": -0.005, "ci95_high": 0.01, "num_contents": 10}
    aggregates = {
        "early_macro": {"kld_improvement": positive, "sim_improvement": positive, "cc_improvement": positive},
        "per_horizon": {
            "0.5": {"kld_improvement": positive, "sim_improvement": positive, "cc_improvement": positive},
            "1": {"kld_improvement": positive, "sim_improvement": positive, "cc_improvement": positive},
            "3": {"kld_improvement": neutral, "sim_improvement": neutral, "cc_improvement": neutral},
            "5": {"kld_improvement": neutral, "sim_improvement": neutral, "cc_improvement": neutral},
        },
    }

    decision = decide_eh0(aggregates, canonical_start_verified=True, outer_test_used=False)

    assert decision["decision"] == "PASS"
    assert all(decision["checks"].values())


def test_eh0_decision_fails_if_outer_test_was_used() -> None:
    positive = {"mean": 0.1, "ci95_low": 0.01, "ci95_high": 0.2, "num_contents": 10}
    aggregates = {
        "early_macro": {"kld_improvement": positive, "sim_improvement": positive, "cc_improvement": positive},
        "per_horizon": {str(h): {"kld_improvement": positive, "sim_improvement": positive, "cc_improvement": positive} for h in (0.5, 1, 3, 5)},
    }
    assert decide_eh0(aggregates, canonical_start_verified=True, outer_test_used=True)["decision"] == "FAIL"


def test_non_test_manifest_never_returns_outer_test_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"image_id": "P1", "split": "train"},
        {"image_id": "P2", "split": "val"},
        {"image_id": "P3", "split": "gate"},
        {"image_id": "P4", "split": "test"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    selected, access = load_non_test_manifest(manifest)

    assert [row["image_id"] for row in selected] == ["P1", "P2", "P3"]
    assert access == {"manifest_rows": 4, "selected_non_test_rows": 3, "outer_test_metadata_rows": 1, "outer_test_target_rows_loaded": 0}
