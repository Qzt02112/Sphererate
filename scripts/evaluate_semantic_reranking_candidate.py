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

from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned
from scripts.evaluate_semantic_reranking_oracle import (
    _aggregate,
    _evaluate_fusion,
    _paired_bootstrap,
    _prediction_row,
    _samples,
)


def _prediction_rows(
    samples: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    *,
    horizons: tuple[float, ...],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        image_id = str(sample["image_id"])
        with np.load(str(sample["fixations_path"])) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        observers = np.unique(fixations[:, 3]).astype(int).tolist()
        for index, horizon in enumerate(horizons):
            rows.append(
                _prediction_row(
                    np.asarray(predictions[image_id][index], dtype=np.float64),
                    sample,
                    horizon_index=index,
                    horizon=horizon,
                    budgets=budgets,
                    fixations=fixations,
                    observers=observers,
                )
            )
    return rows


def _summary(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['label']} Validation Audit",
        "",
        f"Decision: **{result['decision']}**. Evaluation uses Salient360 val only; gate/test target loads are zero.",
        "",
        "| Method | KLD ↓ | CC ↑ | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("template", "candidate", "wrong_image", "spherehorizon", "tempsal"):
        row = result["methods"][method]["early_macro"]
        lines.append(
            f"| {method} | {row['kld_sph']:.4f} | {row['cc_sph']:.4f} | "
            f"{row['coverage_5']:.4f} | {row['coverage_10']:.4f} | {row['coverage_20']:.4f} |"
        )
    lines.extend(["", "## Frozen acceptance checks", ""])
    for name, check in result["checks"].items():
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'} — {name}: {check['detail']}")
    lines.extend(["", "## 3 s correct-minus-wrong attribution", ""])
    for metric, values in result["correct_minus_wrong_3s"].items():
        lines.append(
            f"- {metric}: {values['mean']:+.4f} "
            f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}]"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one semantic reranking candidate on val.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    horizons = (0.5, 1.0, 3.0, 5.0)
    primary = {0.5, 1.0, 3.0}
    budgets = (5.0, 10.0, 20.0)
    samples = _samples(
        ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2", "val"
    )
    device = torch.device(args.device)
    checkpoint = ROOT / args.checkpoint
    correct, provenance = _predict_learned(checkpoint, samples, device)
    wrong, wrong_provenance = _predict_learned(
        checkpoint, samples, device, counterfactual="wrong_image"
    )
    rows = {
        "candidate": _prediction_rows(
            samples, correct, horizons=horizons, budgets=budgets
        ),
        "wrong_image": _prediction_rows(
            samples, wrong, horizons=horizons, budgets=budgets
        ),
        "template": _evaluate_fusion(
            samples,
            {str(sample["image_id"]): sample["final"].numpy()[0] for sample in samples},
            {horizon: 0.0 for horizon in horizons},
            horizons=horizons,
            budgets=budgets,
        ),
    }
    methods = {
        name: _aggregate(current, horizons=horizons, primary=primary, budgets=budgets)
        for name, current in rows.items()
    }
    s0 = json.loads((ROOT / "reports/semantic_reranking_s0/results.json").read_text())
    methods["spherehorizon"] = s0["methods"]["spherehorizon"]
    methods["tempsal"] = s0["methods"]["tempsal"]
    candidate = methods["candidate"]["early_macro"]
    strongest_coverage = {
        budget: max(
            methods["spherehorizon"]["early_macro"][f"coverage_{budget:g}"],
            methods["tempsal"]["early_macro"][f"coverage_{budget:g}"],
        )
        for budget in (5.0, 10.0)
    }
    attribution = {
        metric: _paired_bootstrap(
            rows["candidate"],
            rows["wrong_image"],
            horizon=3.0,
            metric=metric,
            iterations=10000,
            seed=args.seed,
        )
        for metric in ("kld_sph", "cc_sph", "coverage_5", "coverage_10")
    }
    checks = {
        "KLD guard": {
            "pass": candidate["kld_sph"] <= 1.10999,
            "detail": f"{candidate['kld_sph']:.5f} <= 1.10999",
        },
        "Cov@5 over strongest reference": {
            "pass": candidate["coverage_5"] > strongest_coverage[5.0],
            "detail": f"{candidate['coverage_5']:.5f} > {strongest_coverage[5.0]:.5f}",
        },
        "Cov@10 over strongest reference": {
            "pass": candidate["coverage_10"] > strongest_coverage[10.0],
            "detail": f"{candidate['coverage_10']:.5f} > {strongest_coverage[10.0]:.5f}",
        },
        "correct-image attribution at 3 s": {
            "pass": attribution["cc_sph"]["ci95_low"] > 0.0
            and attribution["kld_sph"]["ci95_high"] < 0.0,
            "detail": "requires CC CI > 0 and KLD CI < 0 for correct minus wrong image",
        },
    }
    decision = "PASS_AUTHORIZE_CONFIRMATION" if all(x["pass"] for x in checks.values()) else "FAIL_RETAIN_CURRENT_MAINLINE"
    result = {
        "label": args.label,
        "decision": decision,
        "protocol": {
            "split": "Salient360 val",
            "checkpoint": str(checkpoint),
            "provenance": provenance,
            "wrong_image_provenance": wrong_provenance,
            "gate_test_target_loads": 0,
        },
        "methods": methods,
        "correct_minus_wrong_3s": attribution,
        "checks": checks,
    }
    output = ensure_dir(ROOT / args.output)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
