from __future__ import annotations

from pathlib import Path

from sphererate.config import apply_overrides, load_config


NON_FINAL_LOSS_WEIGHTS = (
    "condition_contrastive",
    "static",
    "temporal",
    "temporal_frame_align",
    "temporal_profile",
    "cumulative",
    "temporal_bin_kld",
    "cumulative_bin_kld",
    "temporal_process_tv",
    "cumulative_process_tv",
    "accumulation",
    "memory",
    "residual",
    "teacher_residual_target",
    "teacher_residual_topk",
    "teacher_residual_peak",
    "motion_bin_ce",
    "motion_bin_soft_ce",
)


def test_gate_a_static_batch_is_clean_final_only_and_seed_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/multi360_256x512.yaml")
    batch = load_config(root / "configs/batch_gate_a_static_3seed.yaml")

    assert batch["protocol"] == "P0_image_only"
    assert batch["gate"] == "A-2"
    assert batch["test_status"] == "frozen_not_evaluated"
    assert len(batch["runs"]) == 3

    seeds: set[int] = set()
    invariant_snapshots: list[dict] = []
    for run in batch["runs"]:
        assert run["model"] == "semantic_static"
        cfg = apply_overrides(base, run["overrides"])
        assert "groupdisjoint_v1" in cfg["data"]["processed_name"]
        assert cfg["train"]["selection_metric"] == "macro_kld_sph"
        assert cfg["train"]["selection_mode"] == "min"
        assert cfg["train"]["dataset_balanced_sampling"] is False
        assert cfg["train"]["augment_roll_degrees"] == [0, 60, 120, 180, 240, 300]
        assert cfg["train"]["shuffle_temporal_labels"] is False
        assert int(cfg["train"]["condition_prefix_bins"]) == 0
        assert cfg["train"]["condition_startpoint_channels"] is False
        assert cfg["train"]["validate_process_metrics"] is False
        assert all(float(cfg["loss"][key]) == 0.0 for key in NON_FINAL_LOSS_WEIGHTS)
        assert float(cfg["loss"]["final_kl"]) == 1.0
        assert float(cfg["loss"]["final_cc"]) == 1.0
        assert float(cfg["loss"]["final_nss"]) == 0.1

        seed = int(cfg["project"]["seed"])
        seeds.add(seed)
        assert int(cfg["train"]["data_seed"]) == seed
        assert int(cfg["train"]["sampler_seed"]) == seed
        invariant_snapshots.append(
            {
                "epochs": cfg["train"]["epochs"],
                "batch_size": cfg["train"]["batch_size"],
                "lr": cfg["train"]["lr"],
                "weight_decay": cfg["train"]["weight_decay"],
                "selection_metric": cfg["train"]["selection_metric"],
                "loss": cfg["loss"],
                "model": cfg["model"],
            }
        )

    assert seeds == {20260710, 20260711, 20260712}
    assert invariant_snapshots[1:] == invariant_snapshots[:-1]
