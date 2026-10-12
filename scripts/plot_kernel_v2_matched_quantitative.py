#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/kernel_v2_matched_baselines/results.json"
OUTPUT = ROOT / "reports/kernel_v2_matched_quantitative"

METHODS = (
    "Early template",
    "TempSAL-ERP",
    "GBVS360 + reachable",
    "DeepGaze MSDB + reachable",
    "Pathformer3D",
    "ScanDMM",
    "SphereHorizon",
)
CONTROLS = METHODS[:-1]
LABEL = {
    "Early template": "Early template",
    "TempSAL-ERP": "TempSAL-ERP",
    "GBVS360 + reachable": "GBVS360 + reach.",
    "DeepGaze MSDB + reachable": "MSDB-25 + reach.",
    "Pathformer3D": "Pathformer3D",
    "ScanDMM": "ScanDMM",
    "SphereHorizon": "SphereHorizon",
}
COLOR = {
    "Early template": "#7A7A7A",
    "TempSAL-ERP": "#E69F00",
    "GBVS360 + reachable": "#CC79A7",
    "DeepGaze MSDB + reachable": "#D55E00",
    "Pathformer3D": "#009E73",
    "ScanDMM": "#56B4E9",
    "SphereHorizon": "#2166AC",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def _absolute(axis: plt.Axes, data: dict, key: str, title: str, xlabel: str) -> None:
    y = np.arange(len(METHODS))[::-1]
    values = [
        data["gate"]["methods"][method]["metrics"].get(key)
        if key != "coverage"
        else data["gate"]["methods"][method]["coverage"]["10.0"]
        for method in METHODS
    ]
    for yi, method, value in zip(y, METHODS, values):
        axis.scatter(
            value,
            yi,
            s=31 if method == "SphereHorizon" else 22,
            color=COLOR[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    axis.set_yticks(y, [LABEL[method] for method in METHODS])
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel)
    _clean_axis(axis)


def _paired(axis: plt.Axes, data: dict, kind: str, title: str, xlabel: str) -> None:
    y = np.arange(len(CONTROLS))[::-1]
    for yi, method in zip(y, CONTROLS):
        if kind == "kld_sph":
            record = data["gate"]["spherehorizon_comparisons"][method]["map"]["early_macro"][kind]
        else:
            record = data["gate"]["spherehorizon_comparisons"][method]["coverage"]["10.0"]
        mean, low, high = (float(record[name]) for name in ("mean", "ci95_low", "ci95_high"))
        axis.errorbar(
            mean,
            yi,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            markersize=4.0,
            color=COLOR[method],
            ecolor=COLOR[method],
            elinewidth=1.0,
            capsize=2.0,
            capthick=0.8,
            zorder=3,
        )
    axis.axvline(0.0, color="#333333", linewidth=0.75, linestyle="--")
    axis.set_yticks(y, [LABEL[method] for method in CONTROLS])
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel)
    _clean_axis(axis)


def _reachable(axis: plt.Axes, data: dict) -> None:
    pairs = (
        ("Panoramic-CNN", "Panoramic-CNN + reachable", "Panoramic-CNN"),
        ("BMS360", "BMS360 + reachable", "BMS360"),
        ("GBVS360", "GBVS360 + reachable", "GBVS360"),
        ("DeepGaze MSDB (ERP-adapted)", "DeepGaze MSDB + reachable", "MSDB-25 ERP"),
    )
    y = np.arange(len(pairs))[::-1]
    for yi, (raw_name, gated_name, display) in zip(y, pairs):
        raw = float(data["gate"]["methods"][raw_name]["metrics"]["kld_sph"])
        gated = float(data["gate"]["methods"][gated_name]["metrics"]["kld_sph"])
        axis.annotate(
            "",
            xy=(gated, yi),
            xytext=(raw, yi),
            arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 0.9},
        )
        axis.scatter(raw, yi, s=22, facecolor="white", edgecolor="#777777", linewidth=0.8, zorder=3)
        axis.scatter(gated, yi, s=26, color="#CC79A7", edgecolor="white", linewidth=0.4, zorder=4)
    sphere = float(data["gate"]["methods"]["SphereHorizon"]["metrics"]["kld_sph"])
    axis.axvline(sphere, color=COLOR["SphereHorizon"], linewidth=1.1)
    axis.text(
        sphere + 0.025,
        2.85,
        "SphereHorizon",
        color=COLOR["SphereHorizon"],
        fontsize=6.2,
        rotation=90,
        va="top",
    )
    axis.set_yticks(y, [display for _, _, display in pairs])
    axis.set_xlabel("KLD$_{sph}$ (lower is better)")
    axis.set_title("(c) Reachability helps, but does not close the gap", loc="left", fontweight="bold")
    _clean_axis(axis)


def main() -> None:
    _style()
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.92))
    axis_kld = figure.add_subplot(grid[0, :2])
    axis_cov = figure.add_subplot(grid[0, 2:])
    axis_reach = figure.add_subplot(grid[1, :2])
    axis_kld_ci = figure.add_subplot(grid[1, 2])
    axis_cov_ci = figure.add_subplot(grid[1, 3])

    _absolute(axis_kld, data, "kld_sph", "(a) Early-map accuracy", "KLD$_{sph}$ ↓")
    _absolute(axis_cov, data, "coverage", "(b) Fixed-budget gaze coverage", "10% tile coverage ↑")
    axis_kld.set_xlim(0.95, 2.08)
    axis_cov.set_xlim(0.79, 0.875)
    _reachable(axis_reach, data)
    axis_reach.set_xlim(0.95, 3.18)
    _paired(axis_kld_ci, data, "kld_sph", "(d) Paired KLD gain", "Sphere improvement")
    _paired(axis_cov_ci, data, "coverage", "(e) Paired coverage gain", "Sphere improvement")
    axis_kld_ci.set_xlim(-0.04, 1.08)
    axis_cov_ci.set_xlim(-0.012, 0.065)
    axis_cov_ci.set_yticklabels([])

    stem = OUTPUT / "matched_gate_benchmark"
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            stem.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(figure)

    caption = (
        "SphereHorizon improves early cumulative-map accuracy beyond a train-only start template, "
        "a matched TempSAL adaptation, a reachability-filtered traditional saliency model, and two "
        "same-split scanpath models. Values are three-seed averages on the 13-content historically "
        "seen development gate over 0.5/1/3 s. Error bars are 95% paired content-bootstrap intervals "
        "(10,000 resamples). Reachability radii are calibrated only on the 50 training contents. "
        "The 0.0173 absolute coverage gain over the early template recovers 11.2% of its remaining "
        "uncovered gaze mass (a derived interpretation, not a replacement endpoint). "
        "The ScanDMM/Pathformer3D 0.5-s row is a first-second approximation; corrected ScanDMM "
        "uses an independent declared-start tensor in both training and inference. "
        "No v2 test target is used."
    )
    result = {
        "source": str(INPUT),
        "split": "historically_seen_gate",
        "contents": 13,
        "primary_horizons_s": [0.5, 1.0, 3.0],
        "bootstrap_resamples": 10000,
        "test_target_loads": 0,
        "caption": caption,
        "figure": str(stem.with_suffix(".png")),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "index.md").write_text(
        "# Matched kernel-v2 quantitative figure\n\n"
        "![Matched gate benchmark](matched_gate_benchmark.png)\n\n"
        f"**Caption.** {caption}\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": result["figure"], "index": str(OUTPUT / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
