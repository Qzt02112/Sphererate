#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.baselines.panoramic_cnn import build_official_panoramic_cnn, resize_spherical_density
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir
from scripts.evaluate_eh6_scandmm import _average_rows, _macro, _score


@torch.no_grad()
def _predict(checkpoint: Path, samples: list[dict[str, Any]], device: torch.device) -> dict[str, np.ndarray]:
    state = torch.load(checkpoint, map_location=device)
    model = build_official_panoramic_cnn().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    predictions = {}
    for sample in samples:
        image = sample["image"][:3].unsqueeze(0).to(device)
        image = F.interpolate(
            image,
            size=(int(state["height"]), int(state["width"])),
            mode="bilinear",
            align_corners=False,
        )
        final = F.softplus(model(image))[0]
        final = resize_spherical_density(
            final,
            int(sample["target"].shape[-2]),
            int(sample["target"].shape[-1]),
        ).cpu().numpy()
        predictions[str(sample["image_id"])] = np.repeat(final[None], 4, axis=0).astype(np.float32)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate leak-free published Panoramic-CNN static baseline.")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", default="reports/eh6_panoramic_static_val")
    parser.add_argument("--allow-outer-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_v1")
    args = parser.parse_args()
    standard_protocol = "standard_v3" in str(args.data_root)
    dataset = EarlyHorizonDataset(
        args.data_root,
        split=args.split,
        target_mode="correct",
        allow_outer_test=True if standard_protocol else bool(args.allow_outer_test),
    )
    samples = [dataset[index] for index in range(len(dataset))]
    budgets = (5.0, 10.0, 15.0, 20.0)
    score_seeds = []
    coverage_seeds = {str(value): [] for value in budgets}
    provenance = []
    device = torch.device(args.device)
    for raw_checkpoint in args.checkpoint:
        checkpoint = Path(raw_checkpoint)
        predictions = _predict(checkpoint, samples, device)
        scores, coverages = _score(predictions, samples, budgets)
        score_seeds.append(scores)
        for budget in budgets:
            coverage_seeds[str(budget)].append(coverages[str(budget)])
        state = torch.load(checkpoint, map_location="cpu")
        provenance.append(
            {
                "checkpoint": str(checkpoint),
                "seed": int(state["seed"]),
                "epoch": int(state["epoch"]),
                "val_kld_sph": float(state["val_kld_sph"]),
                "source_commit": state["source_commit"],
            }
        )
    averaged_scores = _average_rows(score_seeds, ("kld_sph", "sim_sph", "cc_sph"))
    averaged_coverages = {
        budget: _average_rows(rows, ("coverage",)) for budget, rows in coverage_seeds.items()
    }
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template, samples, budgets)
    result = {
        "baseline": "Panoramic-CNN 360 static final repeat",
        "tier": "published_static_360_no_leak",
        "split": args.split,
        "training": (
            "from scratch on standard 59-train / 13-val content split"
            if "standard_v3" in str(args.data_root)
            else "from scratch on historical 50-train / 9-val content split"
        ),
        "input": "ERP RGB only",
        "provenance": provenance,
        "method_early_macro": _macro(averaged_scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            budget: _macro(rows, ("coverage",))["coverage"] for budget, rows in averaged_coverages.items()
        },
        "score_rows": averaged_scores,
        "coverage_rows": averaged_coverages,
        "comparisons": {
            "map_vs_early_template": compare_method_rows(
                averaged_scores, template_scores, iterations=10000, seed=20260711
            ),
            "coverage_vs_early_template": {
                budget: compare_coverage_rows(
                    averaged_coverages[budget], template_coverages[budget], iterations=10000, seed=20260711
                )
                for budget in averaged_coverages
            },
        },
        "evaluation_role": (
            "validation_selected_test_report"
            if standard_protocol and args.split == "test"
            else "validation"
            if standard_protocol
            else "legacy_frozen_post_hoc_benchmark"
            if args.split == "test"
            else "legacy_adapter_validation"
        ),
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric = result["method_early_macro"]
    coverage = result["coverage_early_macro"]
    lines = [
        f"# EH6 Panoramic-CNN Static — {args.split}",
        "",
        "Tier: **published static 360 baseline, leak-free retraining**.",
        "",
        "The published spherical CNN is trained from scratch on the common train split to predict final saliency; its final map is repeated at every early horizon.",
        "",
        "| KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 5% coverage | 10% coverage | 15% coverage | 20% coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
        f"{coverage['5.0']:.6f} | {coverage['10.0']:.6f} | {coverage['15.0']:.6f} | {coverage['20.0']:.6f} |",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
