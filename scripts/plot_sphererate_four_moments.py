#!/usr/bin/env python
"""Render SphereRate's real cumulative predictions at the four query horizons.

Content: standard-split TEST image P11 (the same content shown as Target (GT)
in the paper flow diagram) so the predictions and the flow's GT are directly
comparable.  Weights: frozen seed-20260711 ``best.pt`` -- the primary run,
same source as the real-output insets in ``sphererate_overview``.

The four maps are the model's cumulative density  M(x, T_k)  for the query
horizons  T = 0.5 / 1 / 3 / 5 s  (output key ``cumulative``), i.e. exactly the
"Prediction P" of the flowchart, evaluated at the four declared horizons.

Drawing style deliberately mirrors ``plot_sphererate_overview.py`` (ERP frame
muted as background + semi-transparent magma overlay + cyan start cross), so
the four-moment figure and the overview figure look like siblings.

Artifacts (see ``reports/sphererate_four_moments/``):
  * ``P11_cumulative_4h.{png,pdf,svg}``        clean 1x4 grid, in-map time tags
  * ``P11_cumulative_4h_captioned.png``        same grid + baked caption line
  * ``panels/P11_t0.5s.png ... P11_t5s.png``   individually cropped maps
                                               (ready to drop into the flow's
                                               "Prediction P" node)

Note: unlike the overview script, the model input is assembled directly from
the canonical 256x512 ERP frame and the constant declared start (yaw=90,
lat=0), so no processed V3 dataset has to exist on disk.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.dataset import coord_channels
from sphererate.data.early_dataset import _declared_start_geometry
from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint

# Canonical 256x512 ERP frames shared by every early-horizon view.
ERP_DIR = ROOT / "data/processed/salient360_256x512_k6_order/images"
CHECKPOINT = ROOT / "runs/standard_split_v3/v12c_hybrid_dwell_rate_seed20260711/checkpoints/best.pt"
OUTPUT = ROOT / "reports/sphererate_four_moments"
IMAGE_ID = "P11"
HORIZONS = (0.5, 1.0, 3.0, 5.0)
DECLARED_START_YAW_DEG = 90.0
DECLARED_START_LAT_DEG = 0.0

INK = "#24313A"
MUTED = "#5F6E78"
CYAN = "#00D5FF"

_STYLE_CANDIDATES = (
    "Nimbus Roman",
    "Liberation Serif",
    "Times New Roman",
    "STIXGeneral",
    "DejaVu Serif",
)


def _style() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    family = next((name for name in _STYLE_CANDIDATES if name in available), "DejaVu Serif")
    mpl.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "stix",
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    return family


def _relative(values: np.ndarray) -> np.ndarray:
    """Per-map display scaling (identical to the overview figure)."""
    values = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(values, [0.01, 0.995])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def _mute(rgb: np.ndarray) -> np.ndarray:
    """Desaturate the ERP frame behind the heatmap (identical to overview)."""
    luminance = np.tensordot(rgb[..., :3], np.asarray([0.2126, 0.7152, 0.0722]), axes=1)
    gray = np.repeat(luminance[..., None], 3, axis=-1)
    return np.clip(0.20 * rgb + 0.80 * gray, 0.0, 1.0)


def _draw_map(
    axis: plt.Axes,
    rgb: np.ndarray,
    values: np.ndarray | None,
    *,
    start_xy: tuple[float, float] | None,
    tag: str | None = None,
    marker_size: float = 60.0,
    marker_lw: float = 1.8,
    tag_fontsize: float = 10.0,
) -> None:
    if values is None:
        axis.imshow(rgb, aspect="equal")
    else:
        axis.imshow(_mute(rgb), aspect="equal")
        axis.imshow(_relative(values), cmap="magma", vmin=0.0, vmax=1.0, alpha=0.68, aspect="equal")
    if start_xy is not None:
        axis.scatter(
            [start_xy[0]],
            [start_xy[1]],
            marker="+",
            s=marker_size,
            c=CYAN,
            linewidths=marker_lw,
            zorder=10,
        )
    if tag is not None:
        axis.text(
            0.985,
            0.018,
            tag,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=tag_fontsize,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
            zorder=11,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.7)


def _load_inputs() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, tuple[float, float]]:
    """Rebuild the exact model input for P11 without a processed V3 dataset."""
    image_path = ERP_DIR / f"{IMAGE_ID}.png"
    with Image.open(image_path) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    image = torch.cat(
        [torch.from_numpy(rgb).permute(2, 0, 1), coord_channels(height, width)],
        dim=0,
    ).unsqueeze(0)
    row = {
        "image_id": IMAGE_ID,
        "declared_start_yaw_deg": DECLARED_START_YAW_DEG,
        "declared_start_lat_deg": DECLARED_START_LAT_DEG,
    }
    start_geometry = torch.from_numpy(_declared_start_geometry(row, height, width)).unsqueeze(0)
    start_y, start_x = np.unravel_index(np.argmax(start_geometry[0, 1].numpy()), (height, width))
    return rgb, image, start_geometry, (float(start_x), float(start_y))


def _grid(
    rgb: np.ndarray,
    cumulative: np.ndarray,
    start_xy: tuple[float, float],
    *,
    panel_w_in: float,
    caption: str | None,
    tag_fontsize: float,
) -> plt.Figure:
    """Four 2:1 ERP panels laid out side by side in exact inches."""
    n = len(HORIZONS)
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

    for index, horizon in enumerate(HORIZONS):
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
        _draw_map(
            axis,
            rgb,
            cumulative[index],
            start_xy=start_xy,
            tag=f"t = {horizon:g} s",
            tag_fontsize=tag_fontsize,
        )
    if caption is not None:
        figure.text(
            0.5,
            0.04,
            caption,
            transform=figure.transFigure,
            ha="center",
            va="center",
            fontsize=8.0,
            color=MUTED,
        )
    return figure


@torch.no_grad()
def main() -> None:
    _style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)

    rgb, image, start_geometry, start_xy = _load_inputs()
    output = model(image.to(device), start_geometry=start_geometry.to(device))
    cumulative = output["cumulative"][0].float().cpu().numpy()
    assert cumulative.shape == (len(HORIZONS), rgb.shape[0], rgb.shape[1])

    # Display sanity: the high-mass support should spread with the horizon.
    flat = cumulative.reshape(len(HORIZONS), -1)
    support = (flat > flat.max(axis=1, keepdims=True) * 0.05).sum(axis=1)
    print(f"content={IMAGE_ID} split=test seed=20260711")
    print("horizons :", [f"{h:g} s" for h in HORIZONS])
    print("support  :", [int(v) for v in support])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel_dir = OUTPUT / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Clean 1x4 grid (paper body / LaTeX caption).
    figure = _grid(rgb, cumulative, start_xy, panel_w_in=3.2, caption=None, tag_fontsize=10.0)
    figure.savefig(OUTPUT / "P11_cumulative_4h.png", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT / "P11_cumulative_4h.pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT / "P11_cumulative_4h.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    # Same grid with a baked caption line (non-LaTeX uses / quick preview).
    figure = _grid(
        rgb,
        cumulative,
        start_xy,
        panel_w_in=3.2,
        caption=(
            "SphereRate cumulative-density prediction $\\widehat C_t$ on standard-test P11 "
            "at query horizons $t\\in\\{0.5,1,3,5\\}$ s (real seed-20260711 outputs)."
        ),
        tag_fontsize=10.0,
    )
    figure.savefig(OUTPUT / "P11_cumulative_4h_captioned.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    # Individual maps, ready to crop into the flow's "Prediction P" node.
    for horizon, values in zip(HORIZONS, cumulative):
        tag = f"t = {horizon:g} s"
        safe = f"t{horizon:g}s".replace(".", "p")
        figure = plt.figure(figsize=(3.6, 1.8))
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        _draw_map(axis, rgb, values, start_xy=start_xy, tag=tag, tag_fontsize=11.0)
        figure.savefig(panel_dir / f"P11_{safe}.png", bbox_inches="tight", pad_inches=0.0)
        figure.savefig(panel_dir / f"P11_{safe}.pdf", bbox_inches="tight", pad_inches=0.0)
        plt.close(figure)

    (OUTPUT / "index.md").write_text(
        f"# SphereRate four-moment predictions (P11)\n\n"
        f"![cumulative 0.5/1/3/5 s](P11_cumulative_4h.png)\n\n"
        f"- content: standard-split test **P11** (same content as the flow figure's Target/GT)\n"
        f"- weights: frozen seed-20260711 `best.pt`; all four maps are **actual model outputs**\n"
        f"- maps: cumulative density $M(x,T_k)$ at query horizons 0.5 / 1 / 3 / 5 s\n"
        f"- `P11_cumulative_4h.*`: clean 1x4 grid; `panels/`: individually cropped maps\n",
        encoding="utf-8",
    )
    print(f"output -> {OUTPUT}")
    print("panels ->", sorted(p.name for p in panel_dir.glob("*.png")))


if __name__ == "__main__":
    main()
