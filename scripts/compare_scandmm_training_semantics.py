#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy first-target initialization with independent declared-start ScanDMM."
    )
    parser.add_argument("--output-dir", default="reports/scandmm_training_semantics_comparison")
    args = parser.parse_args()
    result = {
        "comparison": "independent_declared_start_minus_legacy_first_target_initializer",
        "positive_difference_means_corrected_is_better": True,
        "splits": {},
        "test_target_loads": 0,
    }
    for split in ("val", "gate"):
        legacy = _load(ROOT / f"reports/kernel_v2_scandmm_retrained_{split}/results.json")
        corrected = _load(
            ROOT / f"reports/kernel_v2_scandmm_declared_start_{split}/results.json"
        )
        if corrected.get("initialization_semantics") != "independent_declared_tensor":
            raise ValueError(f"Corrected {split} report lacks independent-start semantics")
        result["splits"][split] = {
            "legacy_early_macro": legacy["method_early_macro"],
            "corrected_early_macro": corrected["method_early_macro"],
            "legacy_coverage_10pct": legacy["coverage_early_macro"]["10.0"],
            "corrected_coverage_10pct": corrected["coverage_early_macro"]["10.0"],
            "paired_map_difference": compare_method_rows(
                corrected["score_rows"], legacy["score_rows"], iterations=10000, seed=20260711
            ),
            "paired_coverage_10pct_difference": compare_coverage_rows(
                corrected["coverage_rows"]["10.0"],
                legacy["coverage_rows"]["10.0"],
                iterations=10000,
                seed=20260711,
            ),
        }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ScanDMM training-semantics comparison",
        "",
        "Both rows use the same 50-train/9-val split, three seeds, val-only sparse-checkpoint "
        "selection, 200 generated paths per content and the 2° spherical map renderer. The only "
        "intended training-semantic change is replacing the first one-second gaze initializer with "
        "the independent declared start.",
        "",
        "| split | variant | KLD ↓ | SIM ↑ | CC ↑ | 10% coverage ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("val", "gate"):
        row = result["splits"][split]
        for key, label in (("legacy", "legacy first-target init"), ("corrected", "independent declared start")):
            metric = row[f"{key}_early_macro"]
            lines.append(
                f"| {split} | {label} | {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | "
                f"{metric['cc_sph']:.6f} | {row[f'{key}_coverage_10pct']:.6f} |"
            )
        diff = row["paired_map_difference"]["early_macro"]
        cov = row["paired_coverage_10pct_difference"]
        lines += [
            "",
            f"{split} corrected-minus-legacy oriented differences: KLD "
            f"`{_ci(diff['kld_sph'])}`, SIM `{_ci(diff['sim_sph'])}`, CC `{_ci(diff['cc_sph'])}`, "
            f"10% coverage `{_ci(cov)}`.",
            "",
        ]
    lines += [
        "The legacy row is retained only to quantify the effect of the semantic correction. "
        "Only the independent-start row is eligible for the matched benchmark.",
        "",
        "No v2 test target was loaded.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(row: dict) -> str:
    return f"{row['mean']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"


if __name__ == "__main__":
    main()
