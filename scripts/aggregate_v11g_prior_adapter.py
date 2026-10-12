#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("kld_sph", "sim_sph", "cc_sph", "coverage_5", "coverage_10", "coverage_20")


def _paired_bootstrap(reports: list[dict], metric: str, *, iterations: int = 10000) -> dict:
    content_ids = sorted(
        {row["content_id"] for report in reports for row in report["paired_rows"]}
    )
    differences = []
    for content_id in content_ids:
        seed_values = []
        for report in reports:
            rows = [
                row
                for row in report["paired_rows"]
                if row["content_id"] == content_id and row["horizon_s"] in {0.5, 1.0, 3.0}
            ]
            if len(rows) != 3:
                raise ValueError(f"incomplete early rows for {content_id}")
            if metric == "kld_sph":
                seed_values.append(
                    np.mean([row["tempsal"][metric] - row["adapted"][metric] for row in rows])
                )
            else:
                seed_values.append(
                    np.mean([row["adapted"][metric] - row["tempsal"][metric] for row in rows])
                )
        differences.append(float(np.mean(seed_values)))
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20260805)
    sampled = values[rng.integers(0, len(values), size=(int(iterations), len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(sampled, 2.5)),
        "ci95_high": float(np.percentile(sampled, 97.5)),
        "content_count": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the three V11-G adapter seeds.")
    parser.add_argument("--split", choices=("val", "gate"), default="val")
    args = parser.parse_args()
    suffix = "" if args.split == "val" else "_gate"
    roots = [
        Path(f"reports/v11g_tempsal_prior_adapter_seed20260710{suffix}"),
        Path(f"reports/v11g_tempsal_prior_adapter{suffix}"),
        Path(f"reports/v11g_tempsal_prior_adapter_seed20260712{suffix}"),
    ]
    reports = [json.loads((root / "results.json").read_text(encoding="utf-8")) for root in roots]
    methods = {}
    for method in ("tempsal", "adapted"):
        methods[method] = {
            metric: float(np.mean([report["early_macro"][method][metric] for report in reports]))
            for metric in METRICS
        }
    gains = {
        "kld_sph": methods["tempsal"]["kld_sph"] - methods["adapted"]["kld_sph"],
        **{
            metric: methods["adapted"][metric] - methods["tempsal"][metric]
            for metric in METRICS
            if metric != "kld_sph"
        },
    }
    bootstrap = {metric: _paired_bootstrap(reports, metric) for metric in METRICS}
    per_horizon = {}
    for horizon in ("0.5", "1", "3", "5"):
        per_horizon[horizon] = {}
        for method in ("tempsal", "adapted"):
            per_horizon[horizon][method] = {
                metric: float(
                    np.mean([report["per_horizon"][horizon][method][metric] for report in reports])
                )
                for metric in METRICS
            }
    result = {
        "split": args.split,
        "seeds": [20260710, 20260711, 20260712],
        "selected_alphas": [report["selected_alphas"] for report in reports],
        "early_macro": methods,
        "gains_adapted_over_tempsal": gains,
        "paired_content_bootstrap": bootstrap,
        "per_horizon": per_horizon,
    }
    out = Path(f"reports/v11g_tempsal_prior_adapter_3seed{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# V11-G Three-Seed {args.split} Summary",
        "",
        "| Method | KLD | SIM | CC | Cov@5 | Cov@10 | Cov@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("tempsal", "adapted"):
        values = methods[method]
        lines.append(
            f"| {method} | {values['kld_sph']:.4f} | {values['sim_sph']:.4f} | "
            f"{values['cc_sph']:.4f} | {values['coverage_5']:.4f} | "
            f"{values['coverage_10']:.4f} | {values['coverage_20']:.4f} |"
        )
    lines.extend(["", "Gain (adapted over TempSAL; positive is better):", ""])
    lines.append(
        "- " + ", ".join(f"{metric}={value:+.5f}" for metric, value in gains.items())
    )
    lines.extend(["", "Paired 95% content-bootstrap intervals:", ""])
    for metric, values in bootstrap.items():
        lines.append(
            f"- {metric}: {values['mean']:+.5f} "
            f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
        )
    lines.extend(["", f"Selected train-only alphas: `{result['selected_alphas']}`"])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
