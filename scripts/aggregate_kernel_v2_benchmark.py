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


SEEDS = (20260710, 20260711, 20260712)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _candidate(split: str, data_root: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    dataset = EarlyHorizonDataset(data_root, split=split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    budgets = (5.0, 10.0, 20.0)
    score_seeds, coverage_seeds = [], {str(value): [] for value in budgets}
    for seed in SEEDS:
        predictions, _ = _predict_learned(
            Path(f"runs/kernel_v2_candidate_correct_seed{seed}/checkpoints/best.pt"),
            samples,
            device,
        )
        scores, coverages = _score(predictions, samples, budgets)
        score_seeds.append(scores)
        for budget in budgets:
            coverage_seeds[str(budget)].append(coverages[str(budget)])
    return (
        _average_rows(score_seeds, ("kld_sph", "sim_sph", "cc_sph")),
        {key: _average_rows(value, ("coverage",)) for key, value in coverage_seeds.items()},
    )


def _method_summary(scores: list[dict[str, Any]], coverage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "early_macro": _macro(scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage": {key: _macro(rows, ("coverage",))["coverage"] for key, rows in coverage.items()},
    }


def _reported_method(path: str | Path, *, reachable: bool = False) -> dict[str, Any]:
    result = _load(path)
    if reachable:
        return {
            "early_macro": result["reachable_early_macro"],
            "coverage": result["reachable_coverage_early_macro"],
        }
    return {
        "early_macro": result["method_early_macro"],
        "coverage": result["coverage_early_macro"],
    }


def _split_result(split: str, data_root: str, device: torch.device) -> dict[str, Any]:
    candidate_scores, candidate_coverage = _candidate(split, data_root, device)
    temporal = _load(f"reports/kernel_v2_tempsal_erp_{split}/results.json")
    methods = {
        "SphereHorizon-v2": _method_summary(candidate_scores, candidate_coverage),
        "TempSAL-ERP-v2": {
            "early_macro": temporal["method_early_macro"],
            "coverage": temporal["coverage_early_macro"],
        },
    }
    panoramic_path = Path(f"reports/kernel_v2_panoramic_static_{split}/results.json")
    if panoramic_path.exists():
        methods["Panoramic-CNN-v2"] = _reported_method(panoramic_path)
    for traditional in ("bms360", "gbvs360"):
        methods[traditional.upper()] = _reported_method(
            f"reports/kernel_v2_{traditional}_{split}/results.json"
        )
        methods[f"{traditional.upper()} + reachable-final"] = _reported_method(
            f"reports/kernel_v2_reachable_{traditional}_{split}/results.json",
            reachable=True,
        )
    methods["Panoramic-CNN + reachable-final"] = _reported_method(
        f"reports/kernel_v2_reachable_panoramic_cnn_{split}/results.json",
        reachable=True,
    )
    if split == "val":
        scan = _load("reports/kernel_v2_scandmm_val/results.json")
        methods["ScanDMM external checkpoint (diagnostic)"] = {
            "early_macro": scan["method_early_macro"],
            "coverage": scan["coverage_early_macro"],
        }
    for learned, label in (
        ("scandmm_retrained", "ScanDMM-retrained"),
        ("pathformer3d_retrained", "Pathformer3D-retrained"),
    ):
        path = Path(f"reports/kernel_v2_{learned}_{split}/results.json")
        if path.exists():
            methods[label] = _reported_method(path)
    comparisons = {
        "candidate_vs_tempsal_map": compare_method_rows(candidate_scores, temporal["score_rows"]),
        "candidate_vs_tempsal_coverage": {
            key: compare_coverage_rows(candidate_coverage[key], temporal["coverage_rows"][key])
            for key in candidate_coverage
        },
    }
    return {"methods": methods, "comparisons": comparisons}


def _ci(value: dict[str, Any]) -> str:
    return f"{value['mean']:.6f} [{value['ci95_low']:.6f}, {value['ci95_high']:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the kernel-v2 train/val/gate benchmark.")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--output-dir", default="reports/kernel_v2_benchmark")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "protocol": "kernel-v2 frozen train/val/gate benchmark",
        "target_renderer": "unit-sphere chord Gaussian sigma=2deg",
        "outer_test_used_in_this_benchmark": False,
        "val": _split_result("val", args.data_root, device),
        "gate": _split_result("gate", args.data_root, device),
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Kernel-v2 benchmark",
        "",
        "Learned primary methods use the same 50-train/9-val content split. Traditional methods are deterministic; reachable-final uses only a train-calibrated 95% angular envelope. All rows use the same 2° spherical targets, and no v2 test target was loaded.",
    ]
    for split in ("val", "gate"):
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                "| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 10% coverage ↑ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, current in result[split]["methods"].items():
            metric, coverage = current["early_macro"], current["coverage"]
            lines.append(
                f"| {name} | {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | "
                f"{metric['cc_sph']:.6f} | {coverage['10.0']:.6f} |"
            )
        comparison = result[split]["comparisons"]
        macro = comparison["candidate_vs_tempsal_map"]["early_macro"]
        lines.extend(
            [
                "",
                "SphereHorizon-v2 improvement over TempSAL-ERP-v2:",
                "",
                f"- KLD: {_ci(macro['kld_sph'])}",
                f"- SIM: {_ci(macro['sim_sph'])}",
                f"- CC: {_ci(macro['cc_sph'])}",
                f"- 10% coverage: {_ci(comparison['candidate_vs_tempsal_coverage']['10.0'])}",
            ]
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
