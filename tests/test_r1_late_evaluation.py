from __future__ import annotations

import numpy as np
import pytest

from sphererate.evaluation.r1_late import (
    content_oracle_metadata,
    decide_r1,
    select_fusion_alpha,
    validate_r1_splits,
    wrong_start_longitude,
)


def test_r1_rejects_test_split() -> None:
    validate_r1_splits("train", "val")
    with pytest.raises(ValueError, match="test"):
        validate_r1_splits("train", "test")


def test_wrong_start_is_fixed_180_degree_offset() -> None:
    assert wrong_start_longitude(0.0) == -180.0
    assert wrong_start_longitude(90.0) == -90.0
    assert wrong_start_longitude(180.0) == 0.0
    assert wrong_start_longitude(-135.0) == 45.0


def test_alpha_selection_uses_supplied_train_samples_only() -> None:
    area = np.ones((1, 2), dtype=np.float32)
    start = np.array([[0.8, 0.2]], dtype=np.float32)
    content = np.array([[0.2, 0.8]], dtype=np.float32)
    target = np.array([[0.5, 0.5]], dtype=np.float32)

    selected, grid = select_fusion_alpha(
        [(target, start, content)],
        alpha_grid=[0.0, 0.5, 1.0],
        area_weights=area,
    )

    assert selected == 0.5
    assert set(grid) == {"0.000", "0.500", "1.000"}


def test_content_map_is_explicitly_labelled_oracle() -> None:
    metadata = content_oracle_metadata()
    assert metadata["name"] == "content_oracle"
    assert metadata["deployable"] is False
    assert "Salient360" in metadata["source"]


def test_r1_decision_requires_every_preregistered_condition() -> None:
    positive = {"ci95_low": 0.01, "ci95_high": 0.03}
    neutral_cc = {"ci95_low": -0.005, "ci95_high": 0.02}
    comparisons = {
        "fusion_vs_start": {"kld_sph": positive, "sim_sph": positive, "cc_sph": neutral_cc},
        "fusion_vs_wrong_image": {"kld_sph": positive, "sim_sph": positive},
        "start_vs_wrong180": {"kld_sph": positive, "sim_sph": positive},
    }

    passed = decide_r1(selected_alpha=0.5, comparisons=comparisons)
    failed = decide_r1(selected_alpha=1.0, comparisons=comparisons)

    assert passed["decision"] == "PASS"
    assert failed["decision"] == "FAIL"
    assert failed["checks"]["interior_alpha"] is False
