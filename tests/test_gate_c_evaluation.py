from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("evaluate_gate_c", root / "scripts/evaluate_gate_c.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_c_evaluator_refuses_test_split() -> None:
    module = _module()
    assert module.validate_gate_c_split("val") == "val"
    try:
        module.validate_gate_c_split("test")
    except ValueError as exc:
        assert "validation only" in str(exc)
    else:
        raise AssertionError("Gate-C evaluator accepted the frozen test split")


def test_grouped_bootstrap_orients_lower_and_higher_metrics_as_improvement() -> None:
    module = _module()
    candidate = {
        "a1": {"kld_sph": 0.2, "sim_sph": 0.8},
        "a2": {"kld_sph": 0.4, "sim_sph": 0.6},
        "b1": {"kld_sph": 0.1, "sim_sph": 0.9},
    }
    control = {
        "a1": {"kld_sph": 0.4, "sim_sph": 0.6},
        "a2": {"kld_sph": 0.6, "sim_sph": 0.4},
        "b1": {"kld_sph": 0.3, "sim_sph": 0.7},
    }
    groups = {"a1": "a", "a2": "a", "b1": "b"}
    for metric in ("kld_sph", "sim_sph"):
        result = module.grouped_bootstrap_improvement(candidate, control, groups, metric, 1000, 7)
        assert abs(result["oriented_improvement_mean"] - 0.2) < 1e-8
        assert result["ci95_low"] > 0.0
        assert result["num_groups"] == 2


def test_gate_c_decision_requires_proposed_to_beat_every_control() -> None:
    module = _module()
    summaries = {
        "proposed": {"scores": {"kld_sph": 0.20, "sim_sph": 0.80, "cc_sph": 0.70}, "macro_scores": {"kld_sph": 0.25}},
        "static": {"scores": {"kld_sph": 0.30, "sim_sph": 0.70, "cc_sph": 0.60}, "macro_scores": {"kld_sph": 0.35}},
        "ordinary": {"scores": {"kld_sph": 0.25, "sim_sph": 0.75, "cc_sph": 0.65}, "macro_scores": {"kld_sph": 0.30}},
        "random": {"scores": {"kld_sph": 0.24, "sim_sph": 0.76, "cc_sph": 0.66}, "macro_scores": {"kld_sph": 0.29}},
    }
    assert module.gate_c_seed_decision(summaries, added_parameter_fraction=0.01)["pass"] is True
    summaries["random"]["scores"]["kld_sph"] = 0.19
    decision = module.gate_c_seed_decision(summaries, added_parameter_fraction=0.01)
    assert decision["pass"] is False
    assert "random" in decision["failed_controls"]

