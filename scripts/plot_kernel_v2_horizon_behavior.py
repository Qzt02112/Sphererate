#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/kernel_v2_matched_baselines/results.json"
OUTPUT = ROOT / "reports/kernel_v2_horizon_behavior"
HORIZONS = ("0.5", "1", "3", "5")
CONTROLS = ("Early template", "TempSAL-ERP")
LABEL = {"Early template": "Early template", "TempSAL-ERP": "TempSAL-ERP"}
COLOR = {"Early template": "#777777", "TempSAL-ERP": "#D88900"}
MARKER = {"Early template": "o", "TempSAL-ERP": "s"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _record(data: dict[str, Any], control: str, kind: str, horizon: str) -> dict[str, float]:
    comparison = data["gate"]["spherehorizon_comparisons"][control]
    if kind == "kld":
        return comparison["map"]["per_horizon"][horizon]["kld_sph"]
    return comparison["coverage_per_horizon"]["10.0"][horizon]


def _panel(
    axis: plt.Axes,
    data: dict[str, Any],
    *,
    kind: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    x = np.arange(len(HORIZONS), dtype=np.float32)
    axis.axvspan(-0.38, 2.38, color="#EAF1F8", alpha=0.62, zorder=0)
    axis.axvspan(2.62, 3.38, color="#F0F0F0", alpha=0.80, zorder=0)
    axis.axhline(0.0, color="#333333", linewidth=0.7, linestyle="--", zorder=1)
    for control in CONTROLS:
        records = [_record(data, control, kind, horizon) for horizon in HORIZONS]
        mean = np.asarray([float(record["mean"]) for record in records])
        low = np.asarray([float(record["ci95_low"]) for record in records])
        high = np.asarray([float(record["ci95_high"]) for record in records])
        axis.errorbar(
            x,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            label=LABEL[control],
            color=COLOR[control],
            marker=MARKER[control],
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.7,
            zorder=3,
        )
    axis.set_xlim(-0.38, 3.38)
    axis.set_ylim(*ylim)
    axis.set_xticks(x, [f"{horizon} s" for horizon in HORIZONS])
    axis.set_xlabel("Prediction horizon $t$")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.text(1.0, ylim[1] * 0.91, "primary horizons", color="#4F6B83", ha="center", va="top", fontsize=6.2)
    axis.text(3.0, ylim[1] * 0.91, "guard", color="#666666", ha="center", va="top", fontsize=6.2)


def main() -> None:
    _style()
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.05), constrained_layout=True)
    _panel(
        axes[0],
        data,
        kind="kld",
        title="(a) Content correction has an early operating range",
        ylabel=r"KLD gain over control $\uparrow$",
        ylim=(-0.205, 0.16),
    )
    _panel(
        axes[1],
        data,
        kind="coverage",
        title="(b) Tile gain emerges after support expands",
        ylabel=r"Cov@10 gain over control $\uparrow$",
        ylim=(-0.035, 0.09),
    )
    axes[0].legend(loc="lower left", frameon=False, ncol=2, handlelength=1.8, columnspacing=1.2)

    stem = OUTPUT / "horizon_behavior"
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)

    result = {
        "source": str(INPUT.relative_to(ROOT)),
        "split": "historically_seen_gate",
        "primary_horizons_s": [0.5, 1.0, 3.0],
        "guard_horizon_s": 5.0,
        "bootstrap_resamples": 10000,
        "test_target_loads": 0,
        "controls": {
            control: {
                "kld_gain": {
                    horizon: _record(data, control, "kld", horizon) for horizon in HORIZONS
                },
                "coverage_10_gain": {
                    horizon: _record(data, control, "coverage", horizon)
                    for horizon in HORIZONS
                },
            }
            for control in CONTROLS
        },
        "figure": str(stem.with_suffix(".pdf").relative_to(ROOT)),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "index.md").write_text(
        "# Horizon-wise behavior\n\n"
        "![Horizon behavior](horizon_behavior.png)\n\n"
        "Positive values mean that SphereHorizon improves over the named control. "
        "Intervals are paired 95% content-bootstrap intervals.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
