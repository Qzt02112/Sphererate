from __future__ import annotations

from pathlib import Path

from sphererate.config import apply_overrides, load_config


def test_gate_b_pilot_is_clean_actual_only_and_uses_locked_teacher() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/jufe_start_conditioned_clean_256x512.yaml")
    batch = load_config(root / "configs/batch_gate_b_condition_signal_pilot.yaml")
    run = batch["runs"][0]
    cfg = apply_overrides(base, run["overrides"])

    assert batch["protocol"] == "P1_start_conditioned"
    assert batch["gate"] == "B"
    assert batch["test_status"] == "frozen_not_evaluated"
    assert run["model"] == "start_conditioned_light_residual"
    assert "groupdisjoint_v1" in cfg["data"]["processed_name"]
    assert cfg["train"]["condition_startpoint_channels"] is True
    assert cfg["train"]["startpoint_condition_mode"] == "actual"
    assert cfg["train"]["selection_metric"] == "kld_sph"
    assert cfg["model"]["residual_hidden_channels"] == 16
    assert cfg["model"]["teacher_checkpoint"].endswith(
        "gate_a_static_semantic_finalonly_natural_seed20260710/checkpoints/best.pt"
    )
    assert float(cfg["loss"]["condition_contrastive"]) == 0.0
    assert float(cfg["loss"]["temporal"]) == 0.0
    assert float(cfg["loss"]["cumulative"]) == 0.0
    assert float(cfg["loss"]["final_kl"]) == 1.0
    assert float(cfg["loss"]["final_cc"]) == 1.0


def test_gate_b_three_seed_batch_varies_only_seed_and_locked_teacher() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/jufe_start_conditioned_clean_256x512.yaml")
    batch = load_config(root / "configs/batch_gate_b_condition_signal_3seed.yaml")

    assert len(batch["runs"]) == 3
    seeds: set[int] = set()
    invariant_snapshots: list[dict] = []
    for run in batch["runs"]:
        cfg = apply_overrides(base, run["overrides"])
        seed = int(cfg["project"]["seed"])
        seeds.add(seed)
        assert run["model"] == "start_conditioned_light_residual"
        assert cfg["train"]["startpoint_condition_mode"] == "actual"
        assert int(cfg["train"]["data_seed"]) == seed
        assert int(cfg["train"]["sampler_seed"]) == seed
        assert int(cfg["train"]["startpoint_random_seed"]) == seed
        assert f"seed{seed}/checkpoints/best.pt" in cfg["model"]["teacher_checkpoint"]
        invariant_snapshots.append(
            {
                "data": cfg["data"],
                "epochs": cfg["train"]["epochs"],
                "batch_size": cfg["train"]["batch_size"],
                "lr": cfg["train"]["lr"],
                "weight_decay": cfg["train"]["weight_decay"],
                "selection_metric": cfg["train"]["selection_metric"],
                "loss": cfg["loss"],
                "model_without_teacher": {
                    key: value for key, value in cfg["model"].items() if key != "teacher_checkpoint"
                },
            }
        )

    assert seeds == {20260710, 20260711, 20260712}
    assert invariant_snapshots[1:] == invariant_snapshots[:-1]
