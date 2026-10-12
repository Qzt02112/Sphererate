from __future__ import annotations

from sphererate.train import _seed_from_config


def test_data_seed_can_be_fixed_independently_from_project_seed() -> None:
    cfg = {
        "project": {"seed": 20260709},
        "train": {"data_seed": 20260708},
    }

    assert _seed_from_config(cfg, "data_seed") == 20260708
    assert _seed_from_config(cfg, "sampler_seed") == 20260709


def test_seed_defaults_to_project_seed_for_backward_compatibility() -> None:
    cfg = {"project": {"seed": 1234}, "train": {}}

    assert _seed_from_config(cfg, "data_seed") == 1234
    assert _seed_from_config(cfg, "sampler_seed") == 1234
