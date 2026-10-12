#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/codec_tile_prefetch_test/results.json"
OUTPUT = ROOT / "paper/figures/codec_application_tikz.tex"
OUTPUT_VERTICAL = ROOT / "paper/figures/codec_application_vertical_tikz.tex"
METHODS = (
    ("early_template", "Early template", "gray", "dashed"),
    ("tempsal_erp", "TempSAL-ERP", "codecOrange", "solid"),
    ("sphererate", "SphereRate", "codecBlue", "solid"),
    ("gt_attention_oracle", "GT-attention oracle", "black", "dotted"),
)
BUDGETS = (5.0, 10.0, 15.0, 20.0)


def _point(
    x: float,
    y: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    ox: float,
    oy: float,
) -> tuple[float, float]:
    width, height = 6.2, 3.7
    px = ox + width * (x - xlim[0]) / (xlim[1] - xlim[0])
    py = oy + height * (y - ylim[0]) / (ylim[1] - ylim[0])
    return px, py


def _panel(
    data: dict,
    *,
    ox: float,
    oy: float = 0.85,
    panel: str,
    title: str,
    xmetric: str,
    ymetric: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: tuple[float, ...],
    yticks: tuple[float, ...],
    xlabel: str,
    ylabel: str,
) -> list[str]:
    width, height = 6.2, 3.7
    lines = [
        f"\\node[anchor=west,font=\\bfseries] at ({ox:.3f},{oy + 4.23:.3f}) {{{panel}}};",
        f"\\node[anchor=center,font=\\bfseries] at ({ox + width / 2:.3f},{oy + 4.23:.3f}) {{{title}}};",
    ]
    for tick in yticks:
        _, py = _point(xlim[0], tick, xlim, ylim, ox=ox, oy=oy)
        lines.extend(
            [
                f"\\draw[codecGrid] ({ox:.3f},{py:.3f}) -- ({ox + width:.3f},{py:.3f});",
                f"\\node[anchor=east] at ({ox - .10:.3f},{py:.3f}) {{{tick:g}}};",
            ]
        )
    for tick in xticks:
        px, _ = _point(tick, ylim[0], xlim, ylim, ox=ox, oy=oy)
        lines.extend(
            [
                f"\\draw[black!45] ({px:.3f},{oy - .03:.3f}) -- ({px:.3f},{oy + .03:.3f});",
                f"\\node[anchor=north] at ({px:.3f},{oy - .13:.3f}) {{{tick:g}}};",
            ]
        )
    lines.extend(
        [
            f"\\draw[black!70] ({ox:.3f},{oy:.3f}) rectangle ({ox + width:.3f},{oy + height:.3f});",
            f"\\node[anchor=north] at ({ox + width / 2:.3f},{oy - .69:.3f}) {{{xlabel}}};",
            f"\\node[rotate=90,anchor=south] at ({ox - .78:.3f},{oy + height / 2:.3f}) {{{ylabel}}};",
        ]
    )
    for method, _, color, dash in METHODS:
        points = []
        for budget in BUDGETS:
            value = data[method][f"{budget:g}"]
            x = budget if xmetric == "budget" else float(value[xmetric])
            y = float(value[ymetric])
            px, py = _point(x, y, xlim, ylim, ox=ox, oy=oy)
            points.append(f"({px:.3f},{py:.3f})")
        lines.append(
            f"\\draw[{color},line width=.85pt,{dash}] " + " -- ".join(points) + ";"
        )
        for point in points:
            lines.append(
                f"\\filldraw[fill=white,draw={color},line width=.7pt] {point} circle (1.15pt);"
            )
    return lines


def main() -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    data = result["aggregate_all_four_horizons"]
    lines = [
        "\\begin{tikzpicture}[x=1cm,y=1cm,",
        "  every node/.style={font=\\sffamily\\fontsize{6.4}{7.2}\\selectfont},",
        "  codecGrid/.style={draw=black!12,line width=.25pt}]",
        "\\definecolor{codecBlue}{RGB}{0,92,155}",
        "\\definecolor{codecOrange}{RGB}{220,112,32}",
    ]
    lines.extend(
        _panel(
            data,
            ox=1.05,
            panel="(a)",
            title="Attention-weighted reconstruction",
            xmetric="bits_per_pixel",
            ymetric="gaze_psnr_db",
            xlim=(0.26, 0.405),
            ylim=(32.5, 37.8),
            xticks=(0.28, 0.32, 0.36, 0.40),
            yticks=(33.0, 34.0, 35.0, 36.0, 37.0),
            xlabel="Actual payload (bits per pixel)",
            ylabel="Gaze-PSNR (dB)",
        )
    )
    lines.extend(
        _panel(
            data,
            ox=9.12,
            panel="(b)",
            title="High-quality tile gaze capture",
            xmetric="budget",
            ymetric="raw_dwell_coverage",
            xlim=(4.0, 21.0),
            ylim=(0.44, 0.93),
            xticks=(5.0, 10.0, 15.0, 20.0),
            yticks=(0.5, 0.6, 0.7, 0.8, 0.9),
            xlabel="Low-to-high byte budget (\\%)",
            ylabel="Raw dwell coverage",
        )
    )
    # One shared legend, arranged horizontally to minimize plot occlusion.
    legend_x = (2.0, 5.45, 9.25, 12.55)
    for (method, label, color, dash), x in zip(METHODS, legend_x):
        lines.extend(
            [
                f"\\draw[{color},line width=.85pt,{dash}] ({x:.2f},5.63) -- ({x + .55:.2f},5.63);",
                f"\\filldraw[fill=white,draw={color},line width=.7pt] ({x + .275:.3f},5.63) circle (1.15pt);",
                f"\\node[anchor=west] at ({x + .65:.2f},5.63) {{{label}}};",
            ]
        )
    lines.append("\\end{tikzpicture}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vertical = [
        "\\begin{tikzpicture}[x=1cm,y=1cm,",
        "  every node/.style={font=\\sffamily\\fontsize{6.4}{7.2}\\selectfont},",
        "  codecGrid/.style={draw=black!12,line width=.25pt}]",
        "\\definecolor{codecBlue}{RGB}{0,92,155}",
        "\\definecolor{codecOrange}{RGB}{220,112,32}",
    ]
    vertical.extend(
        _panel(
            data,
            ox=1.05,
            oy=6.35,
            panel="(a)",
            title="Attention-weighted reconstruction",
            xmetric="bits_per_pixel",
            ymetric="gaze_psnr_db",
            xlim=(0.26, 0.405),
            ylim=(32.5, 37.8),
            xticks=(0.28, 0.32, 0.36, 0.40),
            yticks=(33.0, 34.0, 35.0, 36.0, 37.0),
            xlabel="Actual payload (bits per pixel)",
            ylabel="Gaze-PSNR (dB)",
        )
    )
    vertical.extend(
        _panel(
            data,
            ox=1.05,
            oy=0.85,
            panel="(b)",
            title="High-quality tile gaze capture",
            xmetric="budget",
            ymetric="raw_dwell_coverage",
            xlim=(4.0, 21.0),
            ylim=(0.44, 0.93),
            xticks=(5.0, 10.0, 15.0, 20.0),
            yticks=(0.5, 0.6, 0.7, 0.8, 0.9),
            xlabel="Low-to-high byte budget (\\%)",
            ylabel="Raw dwell coverage",
        )
    )
    legend_positions = ((0.9, 11.75), (4.1, 11.75), (0.9, 11.28), (4.1, 11.28))
    for (_, label, color, dash), (x, y) in zip(METHODS, legend_positions):
        vertical.extend(
            [
                f"\\draw[{color},line width=.85pt,{dash}] ({x:.2f},{y:.2f}) -- ({x + .55:.2f},{y:.2f});",
                f"\\filldraw[fill=white,draw={color},line width=.7pt] ({x + .275:.3f},{y:.2f}) circle (1.15pt);",
                f"\\node[anchor=west] at ({x + .65:.2f},{y:.2f}) {{{label}}};",
            ]
        )
    vertical.append("\\end{tikzpicture}")
    OUTPUT_VERTICAL.write_text("\n".join(vertical) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(OUTPUT_VERTICAL)


if __name__ == "__main__":
    main()
