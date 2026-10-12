from __future__ import annotations

from pathlib import Path

from sphererate.config import load_config


def test_gate_t0_config_is_analysis_only_and_six_bin_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/gate_t0_temporal_decomposition.yaml")
    assert config["gate"] == "T0"
    assert config["network_training"] == "forbidden"
    assert config["test_status"] == "frozen_not_evaluated"
    assert config["temporal"]["num_bins"] == 6
    assert len(config["temporal"]["fixed_bin_weights"]) == 6
    assert abs(sum(config["temporal"]["fixed_bin_weights"]) - 1.0) < 1e-8
    assert config["temporal"]["monotonic"] == "nonincreasing"
    assert config["decision_rule"]["pass"] == "all_components_true"
