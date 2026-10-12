#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_kernel_v2_benchmark import _candidate
from scripts.evaluate_eh6_scandmm import _macro, _score
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir


META = {
    "Early template": {
        "tier": "train-only prior",
        "input": "start + horizon",
        "task": "cumulative map",
        "training": "50-train aggregation",
        "parameters": "0",
    },
    "SphereHorizon": {
        "tier": "matched learned",
        "input": "image + start + horizon",
        "task": "cumulative map",
        "training": "50 train / 9 val, 3 seeds",
        "parameters": "3.69M (0.91M trainable)",
    },
    "TempSAL-ERP": {
        "tier": "matched learned",
        "input": "image + horizon",
        "task": "cumulative map",
        "training": "50 train / 9 val, 3 seeds",
        "parameters": "3.68M (0.90M trainable)",
    },
    "Panoramic-CNN": {
        "tier": "matched published static",
        "input": "image",
        "task": "final map repeated",
        "training": "50 train / 9 val, 3 seeds",
        "parameters": "2.92M",
    },
    "Panoramic-CNN + reachable": {
        "tier": "start-conditioned static",
        "input": "image + start + horizon",
        "task": "gated final map",
        "training": "same static model + train-only radius",
        "parameters": "2.92M",
    },
    "BMS360": {
        "tier": "deterministic traditional",
        "input": "image",
        "task": "final map repeated",
        "training": "not applicable",
        "parameters": "non-neural",
    },
    "BMS360 + reachable": {
        "tier": "start-conditioned static",
        "input": "image + start + horizon",
        "task": "gated final map",
        "training": "train-only radius",
        "parameters": "non-neural",
    },
    "GBVS360": {
        "tier": "deterministic traditional",
        "input": "image",
        "task": "final map repeated",
        "training": "not applicable",
        "parameters": "non-neural",
    },
    "GBVS360 + reachable": {
        "tier": "start-conditioned static",
        "input": "image + start + horizon",
        "task": "gated final map",
        "training": "train-only radius",
        "parameters": "non-neural",
    },
    "DeepGaze MSDB (ERP-adapted)": {
        "tier": "recent pretrained + train calibration",
        "input": "image",
        "task": "calibrated final map repeated",
        "training": "ICCV 2025 weights frozen + 2 train-only scalars",
        "parameters": "451.64M frozen + 2 scalars",
    },
    "DeepGaze MSDB + reachable": {
        "tier": "start-conditioned recent static",
        "input": "image + start + horizon",
        "task": "gated calibrated final map",
        "training": "same frozen model + train-only calibration/radius",
        "parameters": "451.64M frozen + 2 scalars",
    },
    "Pathformer3D": {
        "tier": "matched scanpath",
        "input": "image + start; generated autoregression",
        "task": "1 Hz scanpath → cumulative map",
        "training": "50 train / 9 val, 3 seeds",
        "parameters": "5.05M",
    },
    "ScanDMM": {
        "tier": "matched scanpath",
        "input": "image + independent declared initial gaze",
        "task": "1 Hz scanpath → cumulative map",
        "training": "50 train / 9 val, 3 seeds; independent start in model and guide",
        "parameters": "4.66M",
    },
    "HAT": {
        "tier": "matched recent scanpath",
        "input": "image + declared start as first history token",
        "task": "1 Hz scanpath → cumulative map",
        "training": "50 train / 9 val, 3 seeds; val-only checkpoint selection",
        "parameters": "42.78M (19.32M trainable)",
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _regular(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = _load(path)
    return data["score_rows"], data["coverage_rows"]


def _reachable(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = _load(path)
    return data["reachable_score_rows"], data["reachable_coverage_rows"]


def _deepgaze(
    path: Path, variant: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = _load(path)[variant]
    return data["score_rows"], data["coverage_rows"]


def _per_horizon_means(
    rows: list[dict[str, Any]], metrics: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    horizons = sorted({float(row["horizon_s"]) for row in rows})
    result: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        current = [row for row in rows if float(row["horizon_s"]) == horizon]
        result[f"{horizon:g}"] = {
            metric: sum(float(row[metric]) for row in current) / len(current)
            for metric in metrics
        }
    return result


def _per_horizon_coverage_comparison(
    candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    horizons = sorted({float(row["horizon_s"]) for row in candidate_rows})
    return {
        f"{horizon:g}": compare_coverage_rows(
            [row for row in candidate_rows if float(row["horizon_s"]) == horizon],
            [row for row in control_rows if float(row["horizon_s"]) == horizon],
            primary_horizons_s=(horizon,),
        )
        for horizon in horizons
    }


def _split_methods(
    split: str, data_root: str, device: torch.device
) -> tuple[dict[str, tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]], list[dict[str, Any]]]:
    dataset = EarlyHorizonDataset(data_root, split=split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    candidate_scores, candidate_coverage = _candidate(split, data_root, device)
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverage = _score(template, samples, (5.0, 10.0, 20.0))
    methods = {
        "Early template": (template_scores, template_coverage),
        "SphereHorizon": (candidate_scores, candidate_coverage),
        "TempSAL-ERP": _regular(ROOT / f"reports/kernel_v2_tempsal_erp_{split}/results.json"),
        "Panoramic-CNN": _regular(ROOT / f"reports/kernel_v2_panoramic_static_{split}/results.json"),
        "Panoramic-CNN + reachable": _reachable(
            ROOT / f"reports/kernel_v2_reachable_panoramic_cnn_{split}/results.json"
        ),
        "BMS360": _regular(ROOT / f"reports/kernel_v2_bms360_{split}/results.json"),
        "BMS360 + reachable": _reachable(
            ROOT / f"reports/kernel_v2_reachable_bms360_{split}/results.json"
        ),
        "GBVS360": _regular(ROOT / f"reports/kernel_v2_gbvs360_{split}/results.json"),
        "GBVS360 + reachable": _reachable(
            ROOT / f"reports/kernel_v2_reachable_gbvs360_{split}/results.json"
        ),
        "DeepGaze MSDB (ERP-adapted)": _deepgaze(
            ROOT / f"reports/kernel_v2_deepgaze_msdb_{split}/results.json", "train_calibrated"
        ),
        "DeepGaze MSDB + reachable": _deepgaze(
            ROOT / f"reports/kernel_v2_deepgaze_msdb_{split}/results.json",
            "train_calibrated_reachable",
        ),
        "Pathformer3D": _regular(
            ROOT / f"reports/kernel_v2_pathformer3d_retrained_{split}/results.json"
        ),
    }
    scan = ROOT / f"reports/kernel_v2_scandmm_declared_start_{split}/results.json"
    if scan.exists():
        methods["ScanDMM"] = _regular(scan)
    hat = ROOT / f"reports/kernel_v2_hat_{split}/results.json"
    if hat.exists():
        methods["HAT"] = _regular(hat)
    return methods, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the paper-grade matched kernel-v2 benchmark.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--output-dir", default="reports/kernel_v2_matched_baselines")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "protocol": "kernel_v2_matched_baselines_v1",
        "primary_horizons_s": [0.5, 1.0, 3.0],
        "map_adapter": "sphere_chord_sigma_2deg",
        "reachable_final": "train-only dwell-weighted q95 hard great-circle support",
        "test_target_loads": 0,
        "metadata": META,
    }
    for split in ("val", "gate"):
        methods, _ = _split_methods(split, args.data_root, torch.device(args.device))
        candidate_scores, candidate_coverage = methods["SphereHorizon"]
        summaries, comparisons = {}, {}
        for name, (scores, coverages) in methods.items():
            summaries[name] = {
                "metrics": _macro(scores, ("kld_sph", "sim_sph", "cc_sph")),
                "coverage": {
                    budget: _macro(rows, ("coverage",))["coverage"]
                    for budget, rows in coverages.items()
                },
                "per_horizon": {
                    "metrics": _per_horizon_means(
                        scores, ("kld_sph", "sim_sph", "cc_sph")
                    ),
                    "coverage": {
                        budget: _per_horizon_means(rows, ("coverage",))
                        for budget, rows in coverages.items()
                    },
                },
            }
            if name != "SphereHorizon":
                comparisons[name] = {
                    "map": compare_method_rows(candidate_scores, scores),
                    "coverage": {
                        budget: compare_coverage_rows(candidate_coverage[budget], coverages[budget])
                        for budget in candidate_coverage
                    },
                    "coverage_per_horizon": {
                        budget: _per_horizon_coverage_comparison(
                            candidate_coverage[budget], coverages[budget]
                        )
                        for budget in candidate_coverage
                    },
                }
        result[split] = {"methods": summaries, "spherehorizon_comparisons": comparisons}
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kernel-v2 matched baseline benchmark",
        "",
        "All learned methods use the same 50-train/9-val content split. Conditionable scanpath models receive the declared start as a model input. No v2 test target is loaded.",
    ]
    for split in ("val", "gate"):
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                "| Method | Tier | Input | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 10% coverage ↑ |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for name, summary in result[split]["methods"].items():
            metric = summary["metrics"]
            lines.append(
                f"| {name} | {META[name]['tier']} | {META[name]['input']} | "
                f"{metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
                f"{summary['coverage']['10.0']:.6f} |"
            )
        lines.extend(["", "Paired SphereHorizon improvement over selected strong controls:", ""])
        for name in (
            "Early template",
            "TempSAL-ERP",
            "GBVS360 + reachable",
            "DeepGaze MSDB + reachable",
            "Pathformer3D",
            "ScanDMM",
            "HAT",
        ):
            if name not in result[split]["spherehorizon_comparisons"]:
                continue
            current = result[split]["spherehorizon_comparisons"][name]
            macro = current["map"]["early_macro"]
            coverage = current["coverage"]["10.0"]
            lines.append(
                f"- {name}: KLD `{macro['kld_sph']['mean']:.4f} [{macro['kld_sph']['ci95_low']:.4f}, {macro['kld_sph']['ci95_high']:.4f}]`; "
                f"SIM `{macro['sim_sph']['mean']:.4f} [{macro['sim_sph']['ci95_low']:.4f}, {macro['sim_sph']['ci95_high']:.4f}]`; "
                f"CC `{macro['cc_sph']['mean']:.4f} [{macro['cc_sph']['ci95_low']:.4f}, {macro['cc_sph']['ci95_high']:.4f}]`; "
                f"10% coverage `{coverage['mean']:.4f} [{coverage['ci95_low']:.4f}, {coverage['ci95_high']:.4f}]`."
            )
    lines.extend(
        [
            "",
            "## Reproducibility metadata",
            "",
            "| Method | Training | Native task | Parameters |",
            "|---|---|---|---:|",
        ]
    )
    for name, meta in META.items():
        if name in {"ScanDMM", "HAT"} and name not in result["val"]["methods"]:
            continue
        lines.append(f"| {name} | {meta['training']} | {meta['task']} | {meta['parameters']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
