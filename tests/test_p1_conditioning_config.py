from __future__ import annotations

from pathlib import Path

from sphererate.config import apply_overrides, load_config


def test_p1_conditioning_pilot_is_matched_and_val_only() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/jufe_start_conditioned_clean_256x512.yaml")
    batch = load_config(root / "configs/batch_p1_conditioning_pilot.yaml")

    assert batch["protocol"] == "P1_start_conditioned"
    assert batch["gate"] == "P1-B"
    assert batch["test_status"] == "frozen_not_evaluated"
    assert batch["execution_status"] == "stopped_after_learned_start_only_hard_failure"
    assert [run["status"] for run in batch["runs"]] == [
        "completed_formal_control",
        "incomplete_diagnostic_do_not_use",
        "not_executed_after_gate_failure",
        "not_executed_after_gate_failure",
    ]
    assert [run["model"] for run in batch["runs"]] == [
        "learned_start_only",
        "direct_start_conditioned",
        "film_start_conditioned",
        "start_conditioned_light_residual",
    ]

    snapshots = []
    for run in batch["runs"]:
        cfg = apply_overrides(base, run["overrides"])
        assert cfg["project"]["seed"] == 20260710
        assert cfg["train"]["startpoint_condition_mode"] == "actual"
        assert cfg["train"]["condition_startpoint_channels"] is True
        assert cfg["train"]["augment_roll_degrees"] == [0, 60, 120, 180, 240, 300]
        assert cfg["train"]["selection_metric"] == "kld_sph"
        assert cfg["train"]["selection_mode"] == "min"
        assert cfg["loss"]["condition_contrastive"] == 0.0
        assert cfg["loss"]["temporal"] == 0.0
        assert cfg["loss"]["cumulative"] == 0.0
        assert cfg["model"]["teacher_checkpoint"].endswith(
            "gate_a_static_semantic_finalonly_natural_seed20260710/checkpoints/best.pt"
        )
        snapshots.append(
            {
                "data": cfg["data"],
                "epochs": cfg["train"]["epochs"],
                "batch_size": cfg["train"]["batch_size"],
                "lr": cfg["train"]["lr"],
                "weight_decay": cfg["train"]["weight_decay"],
                "loss": cfg["loss"],
                "model": cfg["model"],
            }
        )

    assert snapshots[1:] == snapshots[:-1]
