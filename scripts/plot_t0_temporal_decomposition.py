#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/gate_t0_temporal_decomposition"


def _group_bootstrap(rows: list[dict[str, str]], key: str, bin_index: int, seed: int = 20260711) -> tuple[float, float, float]:
    by_group: dict[str, list[float]] = {}
    for row in rows:
        if int(row["bin"]) == bin_index:
            by_group.setdefault(row["base_image_id"], []).append(float(row[key]))
    values = np.asarray([np.mean(by_group[group]) for group in sorted(by_group)], dtype=np.float64)
    rng = np.random.default_rng(seed + bin_index)
    samples = np.asarray(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(10000)],
        dtype=np.float64,
    )
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    result = json.loads((REPORT / "results.json").read_text(encoding="utf-8"))
    with (REPORT / "per_bin_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#3B6FB6"
    orange = "#D27A32"
    gray = "#777777"
    green = "#3F8C70"
    bins = np.arange(1, 7)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), constrained_layout=True)

    for ax, metric, ylabel in (
        (axes[0, 0], "kld_sph", "KLD$_{sph}$ ↓"),
        (axes[0, 1], "sim_sph", "SIM$_{sph}$ ↑"),
    ):
        for prefix, color, label in (("start", blue, "Start template"), ("content", orange, "Static content")):
            stats = [_group_bootstrap(rows, f"{prefix}_{metric}", index) for index in range(6)]
            mean = np.asarray([item[0] for item in stats])
            low = np.asarray([item[1] for item in stats])
            high = np.asarray([item[2] for item in stats])
            ax.plot(bins, mean, color=color, marker="o", markersize=3.4, linewidth=1.2, label=label)
            ax.fill_between(bins, low, high, color=color, alpha=0.14, linewidth=0)
        ax.set_xlabel("Temporal bin")
        ax.set_ylabel(ylabel)
        ax.set_xticks(bins)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("(a) Both predictors improve over time")
    axes[0, 1].set_title("(b) Content catches up only in late viewing")
    axes[0, 1].legend(frameon=False, loc="lower right", fontsize=6.5)

    ax = axes[1, 0]
    alphas = np.asarray(result["alpha_selection"]["selected_monotonic_alphas"], dtype=float)
    reliability = result["split_half_reliability"]
    rel_mean = np.asarray([reliability[str(index)]["oriented_reliability"]["mean"] for index in range(6)])
    rel_low = np.asarray([reliability[str(index)]["oriented_reliability"]["ci95_low"] for index in range(6)])
    rel_high = np.asarray([reliability[str(index)]["oriented_reliability"]["ci95_high"] for index in range(6)])
    ax.step(bins, alphas, where="mid", color=blue, linewidth=1.4, label="Start weight α")
    ax.scatter(bins, alphas, color=blue, s=13, zorder=3)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Temporal bin")
    ax.set_ylabel("Selected start weight α", color=blue)
    ax.tick_params(axis="y", colors=blue)
    ax.set_xticks(bins)
    ax.spines["top"].set_visible(False)
    twin = ax.twinx()
    twin.plot(bins, rel_mean, color=green, marker="s", markersize=3.0, linewidth=1.1, label="Residual reliability")
    twin.fill_between(bins, rel_low, rel_high, color=green, alpha=0.13, linewidth=0)
    twin.set_ylabel("Same − different residual cosine", color=green)
    twin.tick_params(axis="y", colors=green)
    twin.spines["top"].set_visible(False)
    ax.set_title("(c) Start weight drops only in the final bin")

    ax = axes[1, 1]
    labels = ["Start-only", "Temporal\nstart template", "Reversed", "Shuffled", "Wrong image"]
    keys = [
        "monotonic_vs_start_only",
        "monotonic_vs_temporal_start_template",
        "monotonic_vs_reversed",
        "monotonic_vs_shuffled",
        "monotonic_vs_wrong_image",
    ]
    values = [result["paired_improvement"][key]["kld_sph"] for key in keys]
    y = np.arange(len(values))[::-1]
    mean = np.asarray([item["oriented_improvement_mean"] for item in values])
    low = np.asarray([item["ci95_low"] for item in values])
    high = np.asarray([item["ci95_high"] for item in values])
    colors = [blue, gray, orange, orange, green]
    ax.axvline(0.0, color="#555555", linewidth=0.7)
    for index in range(len(values)):
        ax.errorbar(
            mean[index],
            y[index],
            xerr=[[mean[index] - low[index]], [high[index] - mean[index]]],
            fmt="o",
            color=colors[index],
            markersize=3.8,
            capsize=2.0,
            linewidth=1.0,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Oriented ΔKLD$_{sph}$ (positive is better)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(d) Content gain over the stronger template is uncertain")

    fig.suptitle("T0 temporal decomposition: ordered fusion improves weak baselines but fails the locked mechanism gate", fontsize=9)
    for suffix in ("pdf", "svg", "png"):
        path = REPORT / f"temporal_decomposition_overview.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
