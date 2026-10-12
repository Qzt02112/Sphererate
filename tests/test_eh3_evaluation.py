from __future__ import annotations

from sphererate.evaluation.early_model import compare_method_rows, decide_eh3


def _rows(candidate: float, control: float, metric: str) -> tuple[list[dict], list[dict]]:
    candidate_rows = []
    control_rows = []
    for content in range(12):
        for horizon in (0.5, 1.0, 3.0, 5.0):
            base = {"content_id": f"P{content}", "horizon_s": horizon}
            candidate_rows.append({**base, metric: candidate})
            control_rows.append({**base, metric: control})
    return candidate_rows, control_rows


def test_comparison_orients_kld_and_similarity_positive() -> None:
    candidate, control = _rows(0.2, 0.4, "kld_sph")
    kld = compare_method_rows(candidate, control, metrics=("kld_sph",), iterations=1000, seed=3)
    candidate, control = _rows(0.8, 0.6, "sim_sph")
    sim = compare_method_rows(candidate, control, metrics=("sim_sph",), iterations=1000, seed=3)

    assert kld["early_macro"]["kld_sph"]["mean"] > 0
    assert sim["early_macro"]["sim_sph"]["mean"] > 0


def test_eh3_decision_requires_all_attribution_and_coverage_checks() -> None:
    positive = {"mean": 0.1, "ci95_low": 0.01, "ci95_high": 0.2, "num_contents": 12}
    neutral_cc = {"mean": 0.0, "ci95_low": -0.005, "ci95_high": 0.01, "num_contents": 12}
    comparisons = {}
    for control in (
        "static_final_repeat",
        "early_template",
        "candidate_shuffled_label",
        "wrong_time",
        "wrong_start",
        "wrong_image",
        "capacity_non_temporal",
    ):
        comparisons[control] = {
            "early_macro": {"kld_sph": positive, "sim_sph": positive, "cc_sph": neutral_cc}
        }
    coverage = {"static_final_repeat": positive, "early_template": positive}
    result = decide_eh3(
        comparisons,
        coverage_comparisons=coverage,
        parameter_match=True,
        outer_test_used=False,
    )

    assert result["decision"] == "PASS"
    comparisons["wrong_image"]["early_macro"]["kld_sph"] = {
        "mean": -0.1,
        "ci95_low": -0.2,
        "ci95_high": -0.01,
        "num_contents": 12,
    }
    comparisons["wrong_image"]["early_macro"]["sim_sph"] = comparisons["wrong_image"]["early_macro"]["kld_sph"]
    assert decide_eh3(
        comparisons,
        coverage_comparisons=coverage,
        parameter_match=True,
        outer_test_used=False,
    )["decision"] == "FAIL"

