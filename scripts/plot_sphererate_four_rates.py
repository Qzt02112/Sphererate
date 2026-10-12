#!/usr/bin/env python
"""Render SphereRate's NON-accumulated interval-rate layers r1..r4 for P11.

This is the sibling of ``plot_sphererate_four_moments.py``.  The four-moments
figure plots the *cumulative* density  M(x,T_k)  (the "Prediction P" output).
This figure instead plots the per-interval dwell RATES
    r_i(x) = Softmax_sph( alpha_i [U + rho_i] - R_i ),
one independent map per scale -- i.e. exactly the four "current-moment" maps
the paper flow shows as ``r1, r2, r3, r4`` feeding the accumulation box
M_k = sum_{i<=k} dt_i r_i.  No panel contains mass from an earlier panel.

Content: standard-split TEST image P11 (the same content as the flow figure's
Target/GT).  Weights: frozen seed-20260711 ``best.pt``.  Drawing style mirrors
``plot_sphererate_overview.py`` and the four-moments figure (muted ERP frame +
semi-transparent magma overlay + cyan start cross + in-map time tag).

Artifacts (see ``reports/sphererate_four_rates/``):
  * ``P11_interval_rates_4h.{png,pdf,svg}``    clean 1x4 grid
  * ``P11_interval_rates_4h_captioned.png``    same grid + baked caption
  * ``panels/P11_r1_0to0p5s.png ...``          individually cropped maps

The cumulative figure in ``reports/sphererate_four_moments/`` is intentionally
left untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Reuse the exact helpers (and model-input pipeline) of the cumulative figure
# so the two figures are pixel-identical siblings apart from the plotted layer.
from plot_sphererate_four_moments import (  # noqa: E402
    CHECKPOINT,
    IMAGE_ID,
    MUTED,
    _draw_map,
    _load_inputs,
    _style,
)
from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint  # noqa: E402

OUTPUT = ROOT / "reports/sphererate_four_rates"

# Interval of each scale, aligned with HORIZONS = (0.5, 1.0, 3.0, 5.0).
INTERVALS = ((0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 5.0))


def _tag(index: int) -> str:
    lo, hi = INTERVALS[index]
    return rf"$r_{index + 1}$  {lo:g}--{hi:g} s"


def _grid(rgb: np.ndarray, rates: np.ndarray, start_xy: tuple[float, float], *, caption: str | None) -> plt.Figure:
    """Four 2:1 ERP panels, one rate layer each, laid out side by side."""
    n = rates.shape[0]
    panel_w_in = 3.2
    gap_in = 0.14
    side_in = 0.04
    panel_h_in = panel_w_in / 2.0
    top_in = 0.02
    bottom_in = 0.02
    caption_h_in = 0.30 if caption is not None else 0.0

    content_w = 2 * side_in + n * panel_w_in + (n - 1) * gap_in
    content_h = top_in + panel_h_in + bottom_in
    figure_h = content_h + caption_h_in
    figure = plt.figure(figsize=(content_w, figure_h))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for index in range(n):
        x_in = side_in + index * (panel_w_in + gap_in)
        y_in = bottom_in + caption_h_in
        axis = figure.add_axes(
            [
                x_in / content_w,
                y_in / figure_h,
                panel_w_in / content_w,
                panel_h_in / figure_h,
            ]
        )
        _draw_map(axis, rgb, rates[index], start_xy=start_xy, tag=_tag(index), tag_fontsize=10.0)
    if caption is not None:
        figure.text(
            0.5, 0.04, caption, transform=figure.transFigure,
            ha="center", va="center", fontsize=8.0, color=MUTED,
        )
    return figure


@torch.no_grad()
def main() -> None:
    _style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)
    model.eval()

    rgb, image, start_geometry, start_xy = _load_inputs()
    output = model(image.to(device), start_geometry=start_geometry.to(device))
    rates = output["interval_rates"][0].float().cpu().numpy()
    assert rates.shape == (len(INTERVALS), rgb.shape[0], rgb.shape[1])

    # Sanity: rate maps are each their own layer (unit spherical mass, and --
    # unlike the cumulative maps -- a later map does not contain the earlier
    # map's top region).
    height, width = rgb.shape[:2]
    lat = np.pi / 2 - (np.arange(height) + 0.5) * (np.pi / height)
    cosw = np.cos(lat).reshape(height, 1)
    masses = (rates * cosw).sum(axis=(1, 2))

    def topk_overlap(prev: np.ndarray, cur: np.ndarray, kfrac: float = 0.5) -> float:
        k = int(prev.size * kfrac)
        top_prev = set(np.argpartition(prev.ravel(), -k)[-k:].tolist())
        top_cur = set(np.argpartition(cur.ravel(), -k)[-k:].tolist())
        return len(top_prev & top_cur) / k

    print(f"content={IMAGE_ID} split=test seed=20260711 (interval RATES, non-cumulative)")
    print("layer  :", [f"r{i + 1} {lo:g}-{hi:g} s" for i, (lo, hi) in enumerate(INTERVALS)])
    print("mass   :", [f"{m:.4f}" for m in masses])
    print("top50 overlap vs previous layer:", [f"{topk_overlap(rates[i], rates[i + 1]):.3f}" for i in range(3)])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel_dir = OUTPUT / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    figure = _grid(rgb, rates, start_xy, caption=None)
    figure.savefig(OUTPUT / "P11_interval_rates_4h.png", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT / "P11_interval_rates_4h.pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT / "P11_interval_rates_4h.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    figure = _grid(
        rgb,
        rates,
        start_xy,
        caption=(
            "SphereRate per-scale dwell rates $r_i$ (non-cumulative interval layers) on "
            "standard-test P11; the paper's accumulation box "
            "$M_k=\\sum_{i\\leq k}\\Delta t_i\\,r_i$ turns these into the cumulative density "
            "(real seed-20260711 outputs)."
        ),
    )
    figure.savefig(OUTPUT / "P11_interval_rates_4h_captioned.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    for index in range(rates.shape[0]):
        lo, hi = INTERVALS[index]
        safe = f"r{index + 1}_{lo:g}to{hi:g}s".replace(".", "p")
        figure = plt.figure(figsize=(3.6, 1.8))
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        _draw_map(axis, rgb, rates[index], start_xy=start_xy, tag=_tag(index), tag_fontsize=11.0)
        figure.savefig(panel_dir / f"P11_{safe}.png", bbox_inches="tight", pad_inches=0.0)
        figure.savefig(panel_dir / f"P11_{safe}.pdf", bbox_inches="tight", pad_inches=0.0)
        plt.close(figure)

    (OUTPUT / "index.md").write_text(
        "# SphereRate interval-rate layers (P11, non-cumulative)\n\n"
        f"![rates r1..r4](P11_interval_rates_4h.png)\n\n"
        f"- content: standard-split test **P11** (same content as the flow figure's Target/GT)\n"
        f"- weights: frozen seed-20260711 `best.pt`; all four maps are **actual model outputs**\n"
        f"- maps: per-interval dwell RATES $r_i$ (0-0.5/0.5-1/1-3/3-5 s), each layer independent\n"
        f"- accumulation (into $M_k$) is left to the paper flow's box $M_k=\\sum \\Delta t_i r_i$\n"
        f"- cumulative version kept at `reports/sphererate_four_moments/`\n",
        encoding="utf-8",
    )
    print(f"output -> {OUTPUT}")
    print("panels ->", sorted(p.name for p in panel_dir.glob("*.png")))


if __name__ == "__main__":
    main()
