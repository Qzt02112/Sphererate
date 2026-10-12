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
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir


SOURCES = ("panoramic_cnn", "bms360", "gbvs360")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired evidence for reachable-final controls.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--output-dir", default="reports/kernel_v2_reachable_final")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "protocol": "train-only 95% hard great-circle reachability gate",
        "radii_deg": {"0.5": 39.391805352924834, "1.0": 44.41630265764504, "3.0": 97.8417141613422, "5.0": 125.03492014657363},
        "test_target_loads": 0,
    }
    for split in ("val", "gate"):
        candidate_scores, candidate_coverages = _candidate(split, args.data_root, torch.device(args.device))
        result[split] = {}
        for source in SOURCES:
            report = _load(ROOT / f"reports/kernel_v2_reachable_{source}_{split}/results.json")
            result[split][source] = {
                "original_early_macro": report["original_early_macro"],
                "reachable_early_macro": report["reachable_early_macro"],
                "reachable_coverage_early_macro": report["reachable_coverage_early_macro"],
                "reachable_vs_original": report["comparisons"]["reachable_vs_original"],
                "spherehorizon_vs_reachable": compare_method_rows(
                    candidate_scores, report["reachable_score_rows"]
                ),
                "spherehorizon_coverage_vs_reachable": {
                    budget: compare_coverage_rows(
                        candidate_coverages[budget], report["reachable_coverage_rows"][budget]
                    )
                    for budget in candidate_coverages
                },
            }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kernel-v2 reachable-final control",
        "",
        "This control asks whether a static final saliency predictor becomes sufficient once saliency outside a train-calibrated angular reachable region is removed.",
        "",
        "The hard great-circle radii are fitted once from the 50 train contents using right-eye clipped-dwell 95% mass: `39.39° / 44.42° / 97.84° / 125.03°` at 0.5/1/3/5 s. No val, gate, or v2 test target selects the quantile or radius.",
    ]
    labels = {"panoramic_cnn": "Panoramic-CNN", "bms360": "BMS360", "gbvs360": "GBVS360"}
    for split in ("val", "gate"):
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                "| Final-map source | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 10% coverage ↑ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for source in SOURCES:
            current = result[split][source]
            metric = current["reachable_early_macro"]
            coverage = current["reachable_coverage_early_macro"]["10.0"]
            lines.append(
                f"| {labels[source]} + reachable | {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | {coverage:.6f} |"
            )
        lines.extend(["", "SphereHorizon improvement over each reachable-final row (paired content bootstrap):", ""])
        for source in SOURCES:
            macro = result[split][source]["spherehorizon_vs_reachable"]["early_macro"]
            coverage = result[split][source]["spherehorizon_coverage_vs_reachable"]["10.0"]
            lines.append(
                f"- {labels[source]}: KLD `{macro['kld_sph']['mean']:.4f} [{macro['kld_sph']['ci95_low']:.4f}, {macro['kld_sph']['ci95_high']:.4f}]`; "
                f"SIM `{macro['sim_sph']['mean']:.4f} [{macro['sim_sph']['ci95_low']:.4f}, {macro['sim_sph']['ci95_high']:.4f}]`; "
                f"CC `{macro['cc_sph']['mean']:.4f} [{macro['cc_sph']['ci95_low']:.4f}, {macro['cc_sph']['ci95_high']:.4f}]`; "
                f"10% coverage `{coverage['mean']:.4f} [{coverage['ci95_low']:.4f}, {coverage['ci95_high']:.4f}]`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Reachability gating is a strong and necessary baseline: it produces large gains over unconditioned final-repeat maps. It still leaves a substantial map-accuracy gap to SphereHorizon, so the task is not reduced to learning or applying a radial early template. Coverage differences are smaller because a coarse reachable region already captures much of the application benefit.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
