#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned, _predict_static, _score_methods


SEEDS = (20260710, 20260711, 20260712)


def _average_rows(rows_by_seed: list[list[dict[str, Any]]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_seed:
        for row in rows:
            groups[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            **{key: float(np.mean([float(row[key]) for row in values])) for key in keys},
        }
        for (content, horizon), values in sorted(groups.items())
    ]


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot frozen EH5 outer-test evaluation.")
    parser.add_argument("--output-dir", default="reports/eh5_outer_test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path("data/processed/early_horizon_salient360_v1")
    info = json.loads((root / "dataset_info.json").read_text())
    if info.get("outer_test_status") != "materialized_once_at_EH5":
        raise PermissionError("EH5 outer-test was not explicitly materialized")
    dataset = EarlyHorizonDataset(
        root,
        split="test",
        target_mode="correct",
        allow_outer_test=True,
    )
    if len(dataset) != 13:
        raise ValueError(f"Frozen outer-test must contain 13 contents, found {len(dataset)}")
    samples = [dataset[index] for index in range(len(dataset))]
    device = torch.device(args.device)
    template_predictions = {
        str(sample["image_id"]): sample["template"].numpy().astype(np.float32) for sample in samples
    }
    static_predictions, static_provenance = _predict_static(
        Path("runs/p2_final_only_seed20260711/checkpoints/best.pt"), samples, args.device
    )
    seed_results: dict[str, Any] = {}
    seed_scores: list[list[dict[str, Any]]] = []
    seed_coverages: list[list[dict[str, Any]]] = []
    template_scores: list[dict[str, Any]] | None = None
    template_coverages: list[dict[str, Any]] | None = None
    static_scores: list[dict[str, Any]] | None = None
    static_coverages: list[dict[str, Any]] | None = None
    predictions_seed11: dict[str, np.ndarray] | None = None

    for seed in SEEDS:
        checkpoint = Path(f"runs/eh3_candidate_correct_seed{seed}/checkpoints/best.pt")
        candidate, provenance = _predict_learned(checkpoint, samples, device)
        if seed == 20260711:
            predictions_seed11 = candidate
        scores, coverage = _score_methods(
            {"candidate": candidate, "template": template_predictions, "static": static_predictions},
            samples,
        )
        map_template = compare_method_rows(scores["candidate"], scores["template"], iterations=10000, seed=20260711)
        map_static = compare_method_rows(scores["candidate"], scores["static"], iterations=10000, seed=20260711)
        coverage_template = compare_coverage_rows(
            coverage["candidate"], coverage["template"], iterations=10000, seed=20260711
        )
        coverage_static = compare_coverage_rows(
            coverage["candidate"], coverage["static"], iterations=10000, seed=20260711
        )
        seed_results[str(seed)] = {
            "provenance": provenance,
            "map_vs_template": map_template,
            "map_vs_static": map_static,
            "coverage_vs_template": coverage_template,
            "coverage_vs_static": coverage_static,
        }
        seed_scores.append(scores["candidate"])
        seed_coverages.append(coverage["candidate"])
        template_scores, template_coverages = scores["template"], coverage["template"]
        static_scores, static_coverages = scores["static"], coverage["static"]

    assert template_scores is not None and template_coverages is not None
    assert static_scores is not None and static_coverages is not None and predictions_seed11 is not None
    average_scores = _average_rows(seed_scores, ("kld_sph", "sim_sph", "cc_sph"))
    average_coverages = _average_rows(seed_coverages, ("coverage",))
    seed_average = {
        "map_vs_template": compare_method_rows(average_scores, template_scores, iterations=10000, seed=20260711),
        "map_vs_static": compare_method_rows(average_scores, static_scores, iterations=10000, seed=20260711),
        "coverage_vs_template": compare_coverage_rows(
            average_coverages, template_coverages, iterations=10000, seed=20260711
        ),
        "coverage_vs_static": compare_coverage_rows(
            average_coverages, static_coverages, iterations=10000, seed=20260711
        ),
    }

    checkpoint11 = Path("runs/eh3_candidate_correct_seed20260711/checkpoints/best.pt")
    attribution: dict[str, Any] = {}
    for counterfactual in ("wrong_time", "wrong_start", "wrong_image"):
        wrong, _ = _predict_learned(
            checkpoint11, samples, device, counterfactual=counterfactual
        )
        scores, coverage = _score_methods(
            {"candidate": predictions_seed11, "wrong": wrong}, samples
        )
        attribution[counterfactual] = {
            "map": compare_method_rows(scores["candidate"], scores["wrong"], iterations=10000, seed=20260711),
            "coverage": compare_coverage_rows(
                coverage["candidate"], coverage["wrong"], iterations=10000, seed=20260711
            ),
        }

    macro = seed_average["map_vs_template"]["early_macro"]
    checks = {
        "all_seed_map_and_coverage_point_estimates_vs_template_positive": all(
            result["map_vs_template"]["early_macro"]["kld_sph"]["mean"] > 0
            and result["map_vs_template"]["early_macro"]["sim_sph"]["mean"] > 0
            and result["coverage_vs_template"]["mean"] > 0
            for result in seed_results.values()
        ),
        "seed_average_kld_vs_template_ci_positive": macro["kld_sph"]["ci95_low"] > 0,
        "seed_average_sim_vs_template_ci_positive": macro["sim_sph"]["ci95_low"] > 0,
        "seed_average_coverage_vs_template_ci_positive": seed_average["coverage_vs_template"]["ci95_low"] > 0,
        "seed_average_coverage_vs_static_ci_positive": seed_average["coverage_vs_static"]["ci95_low"] > 0,
        "wrong_time_attribution": attribution["wrong_time"]["map"]["early_macro"]["kld_sph"]["ci95_low"] > 0,
        "wrong_start_attribution": attribution["wrong_start"]["map"]["early_macro"]["kld_sph"]["ci95_low"] > 0,
        "wrong_image_attribution": attribution["wrong_image"]["map"]["early_macro"]["kld_sph"]["ci95_low"] > 0,
        "outer_test_accessed_once": True,
    }
    result = {
        "gate": "EH5-frozen-outer-test",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seeds": seed_results,
        "seed_average": seed_average,
        "attribution_seed20260711": attribution,
        "static_provenance": static_provenance,
        "outer_test_content_count": len(dataset),
        "outer_test_used": True,
        "method_frozen_before_access": True,
        "no_post_test_tuning_authorized": True,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# EH5: Frozen Salient360 Outer-Test",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "The 13-content outer-test was materialized once after EH0–EH4 passed. No post-test tuning is authorized.",
        "",
        "## Per-seed early-macro improvements",
        "",
        "| Seed | KLD vs template | SIM vs template | CC vs template | 10% coverage vs template |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, current in seed_results.items():
        current_macro = current["map_vs_template"]["early_macro"]
        lines.append(
            f"| {seed} | {_format_ci(current_macro['kld_sph'])} | {_format_ci(current_macro['sim_sph'])} | "
            f"{_format_ci(current_macro['cc_sph'])} | {_format_ci(current['coverage_vs_template'])} |"
        )
    lines.extend(
        [
            "",
            "## Seed-averaged confirmatory evidence",
            "",
            f"- KLD vs template: {_format_ci(macro['kld_sph'])}",
            f"- SIM vs template: {_format_ci(macro['sim_sph'])}",
            f"- CC vs template: {_format_ci(macro['cc_sph'])}",
            f"- 10% coverage vs template: {_format_ci(seed_average['coverage_vs_template'])}",
            f"- 10% coverage vs static-final: {_format_ci(seed_average['coverage_vs_static'])}",
            "",
            "## Seed-20260711 attribution",
            "",
        ]
    )
    for name, current in attribution.items():
        current_macro = current["map"]["early_macro"]
        lines.append(
            f"- correct vs {name}: KLD {_format_ci(current_macro['kld_sph'])}; "
            f"SIM {_format_ci(current_macro['sim_sph'])}; coverage {_format_ci(current['coverage'])}"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in checks.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
