#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.baselines.volume_time import temporal_volume_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir
from scripts.evaluate_eh6_scandmm import _macro, _score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the official pretrained SaltiNet volume.")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--volume-dir", required=True)
    parser.add_argument("--output-dir", default="reports/eh6_saltinet_pretrained_val")
    parser.add_argument("--allow-outer-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_outer_test:
        raise PermissionError("EH6 outer-test evaluation requires explicit --allow-outer-test")
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_v1",
        split=args.split,
        target_mode="correct",
        allow_outer_test=bool(args.allow_outer_test),
    )
    samples = [dataset[index] for index in range(len(dataset))]
    volume_dir = Path(args.volume_dir)
    expected = {str(sample["image_id"]) for sample in samples}
    found = {path.stem for path in volume_dir.glob("*.npz")}
    if found != expected:
        raise ValueError(f"Volume ID mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    predictions = {}
    for sample in samples:
        image_id = str(sample["image_id"])
        with np.load(volume_dir / f"{image_id}.npz") as data:
            volume = np.asarray(data["volume"], dtype=np.float32)
        predictions[image_id] = temporal_volume_to_cumulative_maps(
            volume,
            horizons_s=sample["horizons_s"].numpy().tolist(),
            duration_s=25.0,
            height=int(sample["target"].shape[-2]),
            width=int(sample["target"].shape[-1]),
        )
    budgets = (5.0, 10.0, 20.0)
    scores, coverages = _score(predictions, samples, budgets)
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template, samples, budgets)
    result: dict[str, Any] = {
        "baseline": "SaltiNet official pretrained",
        "split": args.split,
        "tier": "diagnostic_due_to_training_overlap",
        "source_time_semantics": "12 uniform timestamp-quantized slices over 25-second Salient360 stimulus",
        "method_early_macro": _macro(scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            str(budget): _macro(coverages[str(budget)], ("coverage",))["coverage"] for budget in budgets
        },
        "score_rows": scores,
        "coverage_rows": coverages,
        "comparisons": {
            "map_vs_early_template": compare_method_rows(scores, template_scores, iterations=10000, seed=20260711),
            "coverage_vs_early_template": {
                str(budget): compare_coverage_rows(
                    coverages[str(budget)], template_coverages[str(budget)], iterations=10000, seed=20260711
                )
                for budget in budgets
            },
        },
        "leakage_disclosure": (
            "Official paper reports fine-tuning on 36 of the 40 Salient360 training images. "
            "Exact IDs are not recorded in the repository, so overlap with this content split cannot be excluded."
        ),
        "eligible_for_primary_no_leak_table": False,
        "outer_test_role": "frozen_post_hoc_diagnostic" if args.split == "test" else "adapter_validation",
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric = result["method_early_macro"]
    coverage = result["coverage_early_macro"]
    lines = [
        f"# EH6 SaltiNet Official Pretrained — {args.split}",
        "",
        "Status: **diagnostic only; not eligible for the no-leak primary table**.",
        "",
        "The paper defines 12 timestamp-quantized slices and a 25-second Salient360 stimulus. Slices are accumulated by temporal overlap; 0.5/1 s both lie inside the first coarse slice.",
        "",
        "The official checkpoint was fine-tuned on 36/40 Salient360 training images, but the repository does not identify them. Possible content overlap is therefore disclosed rather than hidden.",
        "",
        "| KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 5% coverage | 10% coverage | 20% coverage |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
        f"{coverage['5.0']:.6f} | {coverage['10.0']:.6f} | {coverage['20.0']:.6f} |",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
