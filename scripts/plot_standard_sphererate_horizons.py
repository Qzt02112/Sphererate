#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/standard_split_primary_test/per_content.csv"
OUTPUT = ROOT / "reports/standard_split_horizon_behavior"
PAPER_FIGURE = ROOT / "paper/figures/sphererate_horizon_behavior.pdf"
HORIZONS = (0.5, 1.0, 3.0, 5.0)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.05,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _read() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with INPUT.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "method": raw["method"],
                    "content_id": raw["content_id"],
                    "horizon_s": float(raw["horizon_s"]),
                    "seed": int(raw["seed"]),
                    **{
                        key: float(raw[key])
                        for key in (
                            "kld_sph",
                            "sim_sph",
                            "cc_sph",
                            "coverage_5",
                            "coverage_10",
                            "coverage_15",
                            "coverage_20",
                        )
                    },
                }
            )
    return rows


def _content_means(rows: list[dict[str, object]], method: str, metric: str, horizon: float) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if row["method"] == method and float(row["horizon_s"]) == float(horizon):
            buckets.setdefault(str(row["content_id"]), []).append(float(row[metric]))
    return {content: float(np.mean(values)) for content, values in buckets.items()}


def _gain(
    rows: list[dict[str, object]],
    baseline: str,
    metric: str,
    horizon: float,
) -> dict[str, float | int]:
    ours = _content_means(rows, "v12c", metric, horizon)
    other = _content_means(rows, baseline, metric, horizon)
    contents = sorted(set(ours) & set(other))
    sign = -1.0 if metric == "kld_sph" else 1.0
    values = np.asarray([sign * (ours[key] - other[key]) for key in contents], dtype=np.float64)
    rng = np.random.default_rng(20260805)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "content_count": len(contents),
    }


def _draw_series(
    axis: plt.Axes,
    records: list[dict[str, float | int]],
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    x = np.arange(len(HORIZONS))
    means = np.asarray([float(item["mean"]) for item in records])
    lows = np.asarray([float(item["ci95_low"]) for item in records])
    highs = np.asarray([float(item["ci95_high"]) for item in records])
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        color=color,
        marker=marker,
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.7,
        label=label,
        zorder=3,
    )


def _decorate(axis: plt.Axes, ylabel: str) -> None:
    axis.axvspan(-0.35, 3.35, color="#EAF2F7", alpha=0.55, zorder=0)
    axis.axhline(0.0, color="#32383D", linewidth=0.7, linestyle="--", zorder=1)
    axis.set_xlim(-0.35, 3.35)
    axis.set_xticks(np.arange(len(HORIZONS)), [f"{value:g} s" for value in HORIZONS])
    axis.set_xlabel("Prediction horizon $t$")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D4DADF", alpha=0.72, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    _style()
    rows = _read()
    result: dict[str, object] = {
        "source": str(INPUT.relative_to(ROOT)),
        "split": "standard_test_13_contents",
        "bootstrap_resamples": 10000,
        "comparisons": {},
    }

    density = {
        baseline: [_gain(rows, baseline, "kld_sph", horizon) for horizon in HORIZONS]
        for baseline in ("early_template", "tempsal_erp")
    }
    coverage = {
        budget: [_gain(rows, "tempsal_erp", f"coverage_{budget}", horizon) for horizon in HORIZONS]
        for budget in (5, 10, 15, 20)
    }
    result["comparisons"] = {"kld": density, "coverage_vs_tempsal": coverage}

    figure, axes = plt.subplots(1, 2, figsize=(7.08, 2.08), constrained_layout=True)
    _draw_series(axes[0], density["early_template"], label="Early template", color="#73777A", marker="o")
    _draw_series(axes[0], density["tempsal_erp"], label="TempSAL-ERP", color="#D18400", marker="s")
    axes[0].set_title("(a) Rate accumulation strengthens density prediction", loc="left", fontweight="bold")
    _decorate(axes[0], r"SphereRate KLD gain $\uparrow$")
    axes[0].legend(loc="upper left", frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)

    colors = {5: "#7B3294", 10: "#2C7FB8", 15: "#41AB5D", 20: "#D95F0E"}
    markers = {5: "o", 10: "s", 15: "^", 20: "D"}
    for budget in (5, 10, 15, 20):
        _draw_series(
            axes[1],
            coverage[budget],
            label=f"Cov@{budget}%",
            color=colors[budget],
            marker=markers[budget],
        )
    axes[1].set_title("(b) Application gains depend on horizon and budget", loc="left", fontweight="bold")
    _decorate(axes[1], r"Coverage gain over TempSAL-ERP $\uparrow$")
    axes[1].legend(loc="upper left", frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"sphererate_horizon_behavior.{suffix}", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(PAPER_FIGURE, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "index.md").write_text(
        "# Standard-test horizon behavior\n\n"
        "![Horizon behavior](sphererate_horizon_behavior.png)\n\n"
        "Positive values favor SphereRate. Error bars are paired 95% content-bootstrap intervals.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
