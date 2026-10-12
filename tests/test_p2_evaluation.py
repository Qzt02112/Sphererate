from __future__ import annotations

from sphererate.evaluation.p2_temporal_aux import p2_gate_decision


def _scores(candidate_delta: float) -> dict:
    controls = {
        image_id: {"kld_sph": 0.5, "sim_sph": 0.4, "cc_sph": 0.3}
        for image_id in ("a", "b", "c")
    }
    candidate = {
        image_id: {
            "kld_sph": values["kld_sph"] - candidate_delta,
            "sim_sph": values["sim_sph"] + candidate_delta,
            "cc_sph": values["cc_sph"],
        }
        for image_id, values in controls.items()
    }
    return {
        "temporal_correct": candidate,
        "final_only": controls,
        "temporal_shuffled": controls,
        "capacity_final_repeat": controls,
    }


def test_p2_gate_requires_correct_temporal_to_beat_every_control() -> None:
    groups = {image_id: image_id for image_id in ("a", "b", "c")}
    assert p2_gate_decision(_scores(0.1), groups, iterations=1000, seed=7)["decision"] == "PASS"

    scores = _scores(0.1)
    scores["capacity_final_repeat"] = {
        image_id: {"kld_sph": 0.2, "sim_sph": 0.8, "cc_sph": 0.3}
        for image_id in groups
    }
    result = p2_gate_decision(scores, groups, iterations=1000, seed=7)
    assert result["decision"] == "FAIL"
    assert result["checks"]["temporal_correct_vs_capacity_final_repeat_kld_sph"] is False

