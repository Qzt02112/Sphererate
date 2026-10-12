#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("reports/eh6_visualizations")
DATA = ROOT / "data"
COLORS = {
    "SphereHorizon": "#1769AA",
    "Early template": "#6B7280",
    "ScanDMM": "#E07A1F",
    "Panoramic-CNN": "#7A5195",
    "SaltiNet": "#2A9D8F",
}
MARKERS = {"SphereHorizon": "o", "Early template": "s", "ScanDMM": "^", "Panoramic-CNN": "D", "SaltiNet": "v"}
LINESTYLES = {"SphereHorizon": "-", "Early template": "--", "ScanDMM": "-.", "Panoramic-CNN": ":", "SaltiNet": ":"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _bootstrap(values_by_content: dict[str, float], seed: int = 20260711) -> tuple[float, float, float]:
    values = np.asarray(list(values_by_content.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _metric_curve(rows: list[dict], metric: str, horizons: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, low, high = [], [], []
    for horizon in horizons:
        current = {str(row["content_id"]): float(row[metric]) for row in rows if float(row["horizon_s"]) == horizon}
        result = _bootstrap(current)
        mean.append(result[0]); low.append(result[1]); high.append(result[2])
    return np.asarray(mean), np.asarray(low), np.asarray(high)


def plot_benchmark() -> None:
    payload = json.loads((DATA / "metric_rows.json").read_text())
    methods = payload["methods"]
    horizons = [float(value) for value in payload["horizons_s"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.82, hspace=0.42, wspace=0.28)
    panels = [
        (axes[0, 0], "kld_sph", "KLD$_{sph}$ ↓", "a"),
        (axes[0, 1], "sim_sph", "SIM$_{sph}$ ↑", "b"),
    ]
    for ax, metric, ylabel, label in panels:
        for name, current in methods.items():
            mean, low, high = _metric_curve(current["scores"], metric, horizons)
            ax.plot(
                horizons, mean, color=COLORS[name], marker=MARKERS[name], linestyle=LINESTYLES[name],
                markersize=3.4, label=name,
            )
            ax.fill_between(horizons, low, high, color=COLORS[name], alpha=0.10, linewidth=0)
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(horizons)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.65)
        ax.text(-0.13, 1.05, f"({label})", transform=ax.transAxes, fontweight="bold", fontsize=8.3)
    axes[0, 0].set_yscale("log")

    ax = axes[1, 0]
    for name, current in methods.items():
        rows = current["coverage"]["10.0"]
        mean, low, high = _metric_curve(rows, "coverage", horizons)
        ax.plot(horizons, mean, color=COLORS[name], marker=MARKERS[name], linestyle=LINESTYLES[name], markersize=3.4)
        ax.fill_between(horizons, low, high, color=COLORS[name], alpha=0.10, linewidth=0)
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("10% tile coverage ↑")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    ax.text(-0.13, 1.05, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    ax = axes[1, 1]
    budgets = [5.0, 10.0, 20.0]
    for name, current in methods.items():
        means, lows, highs = [], [], []
        for budget in budgets:
            rows = current["coverage"][str(budget)]
            by_content: dict[str, list[float]] = {}
            for row in rows:
                if float(row["horizon_s"]) in {0.5, 1.0, 3.0}:
                    by_content.setdefault(str(row["content_id"]), []).append(float(row["coverage"]))
            aggregate = {key: float(np.mean(value)) for key, value in by_content.items()}
            result = _bootstrap(aggregate)
            means.append(result[0]); lows.append(result[1]); highs.append(result[2])
        means = np.asarray(means); lows = np.asarray(lows); highs = np.asarray(highs)
        ax.plot(budgets, means, color=COLORS[name], marker=MARKERS[name], linestyle=LINESTYLES[name], markersize=3.4)
        ax.fill_between(budgets, lows, highs, color=COLORS[name], alpha=0.10, linewidth=0)
    ax.set_xlabel("Tile budget (%)")
    ax.set_ylabel("Early-macro coverage ↑")
    ax.set_xticks(budgets)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    ax.text(-0.13, 1.05, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Frozen EH6 benchmark: map accuracy and tile-prefetch coverage", y=0.975, fontsize=9.2, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.3,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"eh6_benchmark_curves.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_qualitative(path: Path, role: str, improvement: float) -> None:
    with np.load(path) as data:
        rgb = data["rgb"]
        horizons = data["horizons_s"]
        stacks = {
            "GT": data["gt"],
            "Early template": data["template"],
            "SphereHorizon": data["spherehorizon"],
            "ScanDMM": data["scandmm"],
            "Panoramic-CNN": data["panoramic"],
            "SaltiNet*": data["saltinet"],
        }
    image_id = path.stem.split("_", 1)[1]
    values = np.concatenate([np.log10(np.maximum(value, 1e-8)).ravel() for value in stacks.values()])
    vmin, vmax = float(np.quantile(values, 0.04)), float(np.quantile(values, 0.997))
    # Every raster is an equirectangular panorama (H:W = 1:2).  Keep square
    # pixels in the rendered panels; ``aspect="auto"`` would stretch maps to
    # fill the GridSpec cells and visually distort spherical locations.
    fig = plt.figure(figsize=(10.6, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(4, 7, width_ratios=[1.18, 1, 1, 1, 1, 1, 1])
    image_ax = fig.add_subplot(grid[0, 0])
    image_ax.imshow(rgb, aspect="equal")
    image_ax.set_box_aspect(rgb.shape[0] / rgb.shape[1])
    image_ax.set_title(f"ERP input\n{image_id}", fontweight="bold")
    image_ax.axis("off")
    note_ax = fig.add_subplot(grid[1:, 0])
    note_ax.axis("off")
    note_ax.text(
        0.5,
        0.78,
        f"Selection\n{role.capitalize()} case\n\nDeclared start\npitch 0° / yaw +90°\n\nSaltiNet*\noverlap diagnostic",
        ha="center",
        va="top",
        fontsize=7.1,
        linespacing=1.35,
        color="#374151",
    )
    last = None
    heat_axes = []
    for col, (name, stack) in enumerate(stacks.items(), start=1):
        for row, horizon in enumerate(horizons):
            ax = fig.add_subplot(grid[row, col])
            heat_axes.append(ax)
            last = ax.imshow(
                np.log10(np.maximum(stack[row], 1e-8)),
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_box_aspect(stack[row].shape[0] / stack[row].shape[1])
            if row == 0:
                ax.set_title(name, fontweight="bold" if name == "SphereHorizon" else "normal")
            if col == 1:
                ax.set_ylabel(f"{float(horizon):g} s", rotation=0, labelpad=15, va="center", fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.1 if name == "SphereHorizon" else 0.45)
                spine.set_edgecolor(COLORS["SphereHorizon"] if name == "SphereHorizon" else "#9CA3AF")
    assert last is not None
    colorbar = fig.colorbar(last, ax=heat_axes, location="bottom", shrink=0.42, pad=0.04, aspect=35)
    colorbar.set_label("log$_{10}$ spherical density (shared scale)")
    title = (
        f"{role.capitalize()} case by SphereHorizon–ScanDMM early-macro KLD improvement: "
        f"{improvement:+.3f}"
    )
    fig.suptitle(title, fontsize=9.3, fontweight="bold")
    output_stem = ROOT / f"qualitative_{role}_{image_id}"
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    ROOT.mkdir(parents=True, exist_ok=True)
    plot_benchmark()
    selection = json.loads((DATA / "selection.json").read_text())
    for row in selection:
        path = DATA / f"{row['role']}_{row['image_id']}.npz"
        plot_qualitative(path, row["role"], float(row["sphere_vs_scandmm_kld_improvement"]))
    lines = [
        "# EH6 Visualizations",
        "",
        "All ScanDMM panels use the declared-start interface `(pitch=0°, yaw=+90°)`. SaltiNet is marked `*` because its official checkpoint may overlap Salient360 contents and is diagnostic only.",
        "All ERP inputs and saliency maps preserve their native `height:width = 1:2` aspect ratio; no panel-wise stretching is applied.",
        "",
        "Related condition-generalization evidence: [EH4 same-panorama multi-start visualization](../gate_eh4_etdk2/eh4_multistart_P13.png). It uses real ET-DK2 session starts and is kept separate from the canonical-start EH6 model benchmark.",
        "",
        "Related content-attribution evidence: [frozen correct/wrong-image audit](../content_utilization_audit/summary.md). It shows that image dependence grows with horizon, while useful gains over the template peak at 3 s.",
        "",
        "## Quantitative benchmark",
        "",
        "![EH6 curves](eh6_benchmark_curves.png)",
        "",
        "Shaded regions are content-bootstrap 95% confidence intervals over 13 frozen test contents. KLD uses a logarithmic y-axis.",
        "",
        "## Qualitative cases",
        "",
    ]
    for row in selection:
        name = f"qualitative_{row['role']}_{row['image_id']}.png"
        lines.extend(
            [
                f"### {row['role'].capitalize()}: {row['image_id']}",
                "",
                f"Transparent selection by SphereHorizon–ScanDMM early-macro KLD improvement: `{row['sphere_vs_scandmm_kld_improvement']:+.4f}`.",
                "",
                f"![{row['role']} case]({name})",
                "",
            ]
        )
    (ROOT / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"index": str(ROOT / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
