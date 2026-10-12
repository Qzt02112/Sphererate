#!/usr/bin/env python
"""Render a SINGLE plain final-cumulative map for P11 (no corner tag).

The four-moments figure (``plot_sphererate_four_moments.py``) plots the
cumulative density M(x,T_k) at all four query horizons with an in-map
"t = ... s" tag.  Some layouts instead want just the final cumulative density
as one clean map with NO text anywhere.  This script renders exactly that:
the model's ``final`` output == M(x, 5 s) (eq. (1) at k=4, spherically
normalized), drawn in the same muted-ERP-frame + semi-transparent magma +
cyan start-cross style as the four-moments figure.

Content: standard-split TEST image P11 (same content as the flow figure's
Target/GT).  Weights: frozen seed-20260711 ``best.pt``.

Artifacts (written into ``reports/sphererate_four_moments/``):
  * ``P11_cumulative_final.{png,pdf,svg}``   single clean 2:1 cumulative map

The existing four-moment / four-rate grids are intentionally left untouched.
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

# Reuse the exact input pipeline and drawing helpers of the four-moments figure.
from plot_sphererate_four_moments import (  # noqa: E402
    CHECKPOINT,
    IMAGE_ID,
    _draw_map,
    _load_inputs,
    _style,
)
from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint  # noqa: E402

OUTPUT = ROOT / "reports/sphererate_four_moments"


@torch.no_grad()
def main() -> None:
    _style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)
    model.eval()

    rgb, image, start_geometry, start_xy = _load_inputs()
    output = model(image.to(device), start_geometry=start_geometry.to(device))

    # ``final`` == cumulative[:, -1:] == M(x, 5 s), spherically normalized.
    final = output["final"][0, 0].float().cpu().numpy()
    assert final.shape == rgb.shape[:2]

    # Sanity: unit spherical mass, and identical to the 4th cumulative panel.
    height, width = rgb.shape[:2]
    lat = np.pi / 2 - (np.arange(height) + 0.5) * (np.pi / height)
    cosw = np.cos(lat).reshape(height, 1)
    mass = float((final * cosw).sum())
    last = output["cumulative"][0, -1].float().cpu().numpy()
    corr = float(np.corrcoef(final.ravel(), last.ravel())[0, 1])
    print(f"content={IMAGE_ID} split=test seed=20260711")
    print(f"final map : M(x,5s)  shape={final.shape}")
    print(f"mass      : {mass:.4f}   corr(finally, cumulative[-1]) = {corr:.6f}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # One clean 2:1 panel, muted frame + magma + start cross, NO corner tag.
    figure = plt.figure(figsize=(3.6, 1.8))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    _draw_map(axis, rgb, final, start_xy=start_xy, tag=None)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUTPUT / f"P11_cumulative_final.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"output -> {OUTPUT / 'P11_cumulative_final.{png,pdf,svg}'}")


if __name__ == "__main__":
    main()
