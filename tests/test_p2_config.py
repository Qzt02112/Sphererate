from __future__ import annotations

from pathlib import Path

from sphererate.config import apply_overrides, load_config


def test_p2_four_runs_are_architecture_and_budget_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/p2_salient360_temporal_aux_256x512.yaml")
    batch = load_config(root / "configs/batch_p2_temporal_aux_pilot.yaml")
    assert batch["protocol"] == "image_only_temporal_auxiliary"
    assert batch["decision_split"] == "gate"
    assert batch["test_status"] == "frozen_not_evaluated"
    assert len(batch["runs"]) == 4
    assert {run["role"] for run in batch["runs"]} == {
        "final_only", "temporal_correct", "temporal_shuffled", "capacity_final_repeat"
    }

    configs = {run["role"]: apply_overrides(base, run["overrides"]) for run in batch["runs"]}
    reference = configs["final_only"]
    for cfg in configs.values():
        assert cfg["model"] == reference["model"]
        assert cfg["model"]["in_channels"] == 8
        assert cfg["model"]["num_bins"] == 6
        assert cfg["train"]["epochs"] == 30
        assert cfg["train"]["batch_size"] == 4
        assert cfg["train"]["lr"] == reference["train"]["lr"]
        assert cfg["train"]["augment_roll_degrees"] == reference["train"]["augment_roll_degrees"]
        assert cfg["train"]["condition_prefix_bins"] == 0
        assert cfg["train"]["condition_startpoint_channels"] is False
        assert cfg["train"]["shuffle_temporal_labels"] is False

    assert configs["final_only"]["loss"]["temporal"] == 0.0
    for role in ("temporal_correct", "temporal_shuffled", "capacity_final_repeat"):
        assert configs[role]["loss"]["temporal"] == 0.5
    assert configs["temporal_correct"]["train"]["temporal_target_mode"] == "correct"
    assert configs["temporal_shuffled"]["train"]["temporal_target_mode"] == "shuffled"
    assert configs["capacity_final_repeat"]["train"]["temporal_target_mode"] == "final_repeat"

