from __future__ import annotations

from pathlib import Path

from sphererate.config import apply_overrides, load_config


def _snapshot(cfg: dict) -> dict:
    return {
        "data": cfg["data"],
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "sampling": cfg["train"]["dataset_balanced_sampling"],
        "augmentation": cfg["train"]["augment_roll_degrees"],
        "selection_metric": cfg["train"]["selection_metric"],
        "teacher": cfg["model"]["teacher_checkpoint"],
        "model_common": {
            key: value
            for key, value in cfg["model"].items()
            if key not in {"latent_aggregation", "teacher_checkpoint"}
        },
        "loss": cfg["loss"],
    }


def test_gate_c_pilot_is_image_only_and_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/multi360_256x512.yaml")
    batch = load_config(root / "configs/batch_gate_c_marginalization_pilot.yaml")

    assert batch["protocol"] == "P0_image_only"
    assert batch["gate"] == "C-pilot"
    assert batch["test_status"] == "frozen_not_evaluated"
    assert len(batch["runs"]) == 3
    configs = [apply_overrides(base, run["overrides"]) for run in batch["runs"]]
    proposed, ordinary, random = configs

    for cfg in configs:
        assert cfg["model"]["in_channels"] == 8
        assert cfg["model"]["num_bins"] == 3
        assert cfg["model"]["latent_num_conditions"] == 3
        assert cfg["model"]["latent_conditioning"] == "shared_multihead"
        assert float(cfg["model"]["teacher_residual_scale"]) == 4.0
        assert cfg["train"]["condition_prefix_bins"] == 0
        assert cfg["train"]["condition_startpoint_channels"] is False
        assert cfg["train"]["selection_metric"] == "macro_kld_sph"
        assert cfg["train"]["dataset_balanced_sampling"] is False
        assert float(cfg["loss"]["temporal"]) == 2.0
        assert float(cfg["loss"]["final_kl"]) == 1.0
        assert float(cfg["loss"]["final_cc"]) == 1.0
        assert cfg["model"]["teacher_checkpoint"].endswith("seed20260710/checkpoints/best.pt")

    assert proposed["model"]["latent_aggregation"] == "density_marginal"
    assert ordinary["model"]["latent_aggregation"] == "logit_pool"
    assert random["model"]["latent_aggregation"] == "density_marginal"
    assert proposed["train"]["shuffle_temporal_labels"] is False
    assert ordinary["train"]["shuffle_temporal_labels"] is False
    assert random["train"]["shuffle_temporal_labels"] is True

    snapshots = [_snapshot(cfg) for cfg in configs]
    snapshots[1]["model_common"] = snapshots[0]["model_common"]
    snapshots[2]["model_common"] = snapshots[0]["model_common"]
    assert snapshots[1] == snapshots[0]
    assert snapshots[2] == snapshots[0]


def test_gate_c_three_seed_matrix_has_seed_matched_teachers_and_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/multi360_256x512.yaml")
    batch = load_config(root / "configs/batch_gate_c_marginalization_3seed.yaml")
    assert len(batch["runs"]) == 9

    by_seed: dict[int, list[tuple[dict, dict]]] = {}
    for run in batch["runs"]:
        cfg = apply_overrides(base, run["overrides"])
        seed = int(cfg["project"]["seed"])
        by_seed.setdefault(seed, []).append((run, cfg))
        assert int(cfg["train"]["data_seed"]) == seed
        assert int(cfg["train"]["sampler_seed"]) == seed
        assert f"seed{seed}/checkpoints/best.pt" in cfg["model"]["teacher_checkpoint"]

    assert set(by_seed) == {20260710, 20260711, 20260712}
    for runs in by_seed.values():
        assert len(runs) == 3
        names = {run["model"] for run, _ in runs}
        assert names == {"image_only_latent_marginal", "image_only_temporal_aux"}
        assert sum(cfg["train"]["shuffle_temporal_labels"] for _, cfg in runs) == 1
