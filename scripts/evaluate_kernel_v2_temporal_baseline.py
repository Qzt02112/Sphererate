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
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir
from scripts.evaluate_eh6_scandmm import _average_rows, _macro, _score


@torch.no_grad()
def _predict(checkpoint: Path, samples: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state = torch.load(checkpoint, map_location=device)
    model = build_model("semantic_tempsal_temporal", state["config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    predictions = {}
    for sample in samples:
        output = model(sample["image"].unsqueeze(0).to(device))["cumulative"][0]
        predictions[str(sample["image_id"])] = output.cpu().numpy().astype(np.float32)
    return predictions, {
        "checkpoint": str(checkpoint),
        "seed": int(state["config"]["project"]["seed"]),
        "epoch": int(state["epoch"]),
        "parameters": state["parameters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the leak-free v2 TempSAL-ERP cumulative baseline.")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", default="reports/kernel_v2_tempsal_erp_val")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = EarlyHorizonDataset(args.data_root, split=args.split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    device = torch.device(args.device)
    budgets = (5.0, 10.0, 15.0, 20.0)
    score_seeds, coverage_seeds, provenance = [], {str(value): [] for value in budgets}, []
    for raw_checkpoint in args.checkpoint:
        predictions, record = _predict(Path(raw_checkpoint), samples, device)
        scores, coverages = _score(predictions, samples, budgets)
        score_seeds.append(scores)
        for budget in budgets:
            coverage_seeds[str(budget)].append(coverages[str(budget)])
        provenance.append(record)
    averaged_scores = _average_rows(score_seeds, ("kld_sph", "sim_sph", "cc_sph"))
    averaged_coverages = {key: _average_rows(value, ("coverage",)) for key, value in coverage_seeds.items()}
    template_predictions = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template_predictions, samples, budgets)
    result = {
        "baseline": "TempSAL-ERP cumulative adaptation",
        "tier": "primary_task_matched_temporal_map",
        "split": str(args.split),
        "training": (
            "same standard 59-train / 13-val split; validation-selected checkpoints; "
            "no start template and no gaze history"
            if "standard_v3" in str(args.data_root)
            else "historical same-split training; no start template and no gaze history"
        ),
        "provenance": provenance,
        "method_early_macro": _macro(averaged_scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            key: _macro(rows, ("coverage",))["coverage"] for key, rows in averaged_coverages.items()
        },
        "score_rows": averaged_scores,
        "coverage_rows": averaged_coverages,
        "comparisons": {
            "map_vs_early_template": compare_method_rows(averaged_scores, template_scores),
            "coverage_vs_early_template": {
                key: compare_coverage_rows(rows, template_coverages[key]) for key, rows in averaged_coverages.items()
            },
        },
        "test_evaluated": bool(args.split == "test"),
        "test_used_for_selection": False,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric, coverage = result["method_early_macro"], result["coverage_early_macro"]
    lines = [
        f"# TempSAL-ERP cumulative baseline — {args.split}",
        "",
        "Leak-free, task-matched temporal-map adaptation trained on the same v2 split.",
        "",
        "| KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | Cov@5% | Cov@10% | Cov@15% | Cov@20% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
        f"{coverage['5.0']:.6f} | {coverage['10.0']:.6f} | {coverage['15.0']:.6f} | "
        f"{coverage['20.0']:.6f} |",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
