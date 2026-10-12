from __future__ import annotations

from pathlib import Path

from sphererate.config import load_config


def test_gate_r0_config_forbids_training_and_test_access() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/gate_r0_content_interaction.yaml")

    assert config["gate"] == "R0"
    assert config["training"] == "forbidden"
    assert config["test_status"] == "frozen_not_evaluated"
    assert config["fusion"]["alpha_selection_split"] == "train"
    assert config["fusion"]["evaluation_split"] == "val"
    assert config["fusion"]["alpha_grid"][0] == 0.0
    assert config["fusion"]["alpha_grid"][-1] == 1.0
    assert config["wrong_image"]["require_different_group_id"] is True
    assert config["wrong_image"]["one_to_one_permutation"] is True
    assert config["decision_rule"]["pass"] == "at_least_one_fusion_passes_and_residual_wrong_image_control_passes"
