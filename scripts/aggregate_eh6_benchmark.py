#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from scripts.evaluate_eh3_gate import _predict_learned
from scripts.evaluate_eh6_scandmm import _average_rows, _macro, _score


def _format_ci(value: dict[str, Any]) -> str:
    return f"{value['mean']:.4f} [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen EH6 benchmark results.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default="reports/eh6_benchmark")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_v1",
        split=args.split,
        target_mode="correct",
        allow_outer_test=args.split == "test",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    budgets = (5.0, 10.0, 20.0)
    candidate_scores_by_seed = []
    candidate_coverages_by_budget = {str(value): [] for value in budgets}
    for seed in (20260710, 20260711, 20260712):
        predictions, _ = _predict_learned(
            Path(f"runs/eh3_candidate_correct_seed{seed}/checkpoints/best.pt"),
            samples,
            torch.device(args.device),
        )
        scores, coverages = _score(predictions, samples, budgets)
        candidate_scores_by_seed.append(scores)
        for budget in budgets:
            candidate_coverages_by_budget[str(budget)].append(coverages[str(budget)])
    candidate_scores = _average_rows(candidate_scores_by_seed, ("kld_sph", "sim_sph", "cc_sph"))
    candidate_coverages = {
        budget: _average_rows(rows, ("coverage",)) for budget, rows in candidate_coverages_by_budget.items()
    }
    template_predictions = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template_predictions, samples, budgets)
    report_paths = {
        "ScanDMM-declared-start": Path(f"reports/eh6_scandmm_declaredstart_{args.split}/results.json"),
        "Panoramic-CNN-static": Path(f"reports/eh6_panoramic_static_{args.split}/results.json"),
        "SaltiNet-pretrained": Path(f"reports/eh6_saltinet_pretrained_{args.split}/results.json"),
    }
    methods: dict[str, dict[str, Any]] = {
        "SphereHorizon": {
            "tier": "proposed",
            "eligible": True,
            "scores": candidate_scores,
            "coverage": candidate_coverages,
        },
        "Early-template": {
            "tier": "internal_nonlearned",
            "eligible": True,
            "scores": template_scores,
            "coverage": template_coverages,
        },
    }
    for name, path in report_paths.items():
        current = json.loads(path.read_text())
        methods[name] = {
            "tier": current["tier"],
            "eligible": bool(current.get("eligible_for_primary_no_leak_table", True)),
            "scores": current["score_rows"],
            "coverage": current["coverage_rows"],
        }
    table = {}
    comparisons = {}
    for name, current in methods.items():
        table[name] = {
            **_macro(current["scores"], ("kld_sph", "sim_sph", "cc_sph")),
            **{
                f"coverage_{budget}": _macro(current["coverage"][budget], ("coverage",))["coverage"]
                for budget in ("5.0", "10.0", "20.0")
            },
            "tier": current["tier"],
            "eligible": current["eligible"],
        }
        if name != "SphereHorizon":
            comparisons[name] = {
                "map": compare_method_rows(
                    candidate_scores, current["scores"], iterations=10000, seed=20260711
                ),
                "coverage_10": compare_coverage_rows(
                    candidate_coverages["10.0"], current["coverage"]["10.0"], iterations=10000, seed=20260711
                ),
            }
    result = {
        "gate": "EH6-frozen-external-baseline-benchmark",
        "status": "IN_PROGRESS",
        "split": args.split,
        "table": table,
        "candidate_improvements": comparisons,
        "completed_categories": {
            "published_static_360_no_leak": True,
            "scanpath_secondary_declared_start": True,
            "official_temporal_volume_overlap_diagnostic": True,
            "task_matched_temporal_no_leak": False,
            "pathformer3d": False,
        },
        "remaining_blocker": "Need leak-free task-matched TempSAL-ERP and SaltiNet-style retraining before EH6 PASS.",
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# EH6 Frozen Benchmark — {args.split}",
        "",
        "Status: **IN PROGRESS**. Published static and declared-start scanpath comparisons are complete; leak-free task-matched temporal baselines remain.",
        "",
        "## Early-macro results",
        "",
        "| Method | Tier | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 5% cov. | 10% cov. | 20% cov. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in table.items():
        marker = "" if row["eligible"] else " (overlap diagnostic)"
        lines.append(
            f"| {name} | {row['tier']}{marker} | {row['kld_sph']:.4f} | {row['sim_sph']:.4f} | "
            f"{row['cc_sph']:.4f} | {row['coverage_5.0']:.4f} | {row['coverage_10.0']:.4f} | {row['coverage_20.0']:.4f} |"
        )
    lines.extend(["", "## SphereHorizon improvements", ""])
    for name, current in comparisons.items():
        macro = current["map"]["early_macro"]
        lines.append(
            f"- vs {name}: KLD {_format_ci(macro['kld_sph'])}; SIM {_format_ci(macro['sim_sph'])}; "
            f"10% coverage {_format_ci(current['coverage_10'])}"
        )
    lines.extend(
        [
            "",
            "## Interpretation lock",
            "",
            "- ScanDMM uses the model's initial 3D gaze-point interface with declared pitch=0°, yaw=+90°; random-start runs are deprecated diagnostics.",
            "- ScanDMM is secondary because its output is one gaze point per second and 0.5 s is a first-second approximation.",
            "- SaltiNet official weights are not eligible for the no-leak main table because their exact 36/40 Salient360 training IDs are unavailable.",
            "- EH6 cannot pass until leak-free task-matched temporal baselines are complete.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
