#!/usr/bin/env python3
"""Create an all-vector, directly editable SphereHorizon framework slide.

The slide is intentionally built only from native PowerPoint shapes.  It contains
no embedded raster image, so every label, arrow, heatmap cell, panorama element,
and tile can be edited independently.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pptx" / "SphereHorizon_editable_framework.pptx"
ACTUAL_DIR = ROOT / "reports" / "spherehorizon_overview" / "assets" / "actual_p10_seed20260711"
ERP_SOURCE = ACTUAL_DIR / "erp_input.png"
ERP_ALIGNED = ACTUAL_DIR / "erp_aligned_center.png"
ACTUALS_NPZ = ACTUAL_DIR / "actuals.npz"

SLIDE_W = 16.0
SLIDE_H = 6.0
# Logical coordinates above stay 16:6, but the physical slide is 12:4.5.  This
# increases the effective type size when LaTeX scales the figure to text width.
GEOM_SCALE = 0.75
TYPE_SCALE = 1.0

FONT = "Liberation Sans"
MATH_FONT = "DejaVu Sans"

WHITE = "FFFFFF"
INK = "17212B"
MUTED = "5F6B76"
LIGHT_TEXT = "7C8792"
LINE = "8A949D"
PANEL = "F5F7F8"
PANEL_2 = "EDF1F3"

TEAL = "25877F"
TEAL_DARK = "176A65"
TEAL_LIGHT = "E6F4F1"
TEAL_MID = "A6D8D2"

GOLD = "B78314"
GOLD_DARK = "8F6510"
GOLD_LIGHT = "FFF4D3"
GOLD_MID = "E9C760"

BLUE = "1859A9"
BLUE_DARK = "123F78"
BLUE_LIGHT = "E8F1FC"

CYAN = "00AFC7"
ORANGE = "DA842A"
ORANGE_LIGHT = "F5C18C"
RED = "C54C40"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def blend(c0: str, c1: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))
    a = tuple(int(c0[i : i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(c1[i : i + 2], 16) for i in (0, 2, 4))
    return "".join(f"{round((1 - alpha) * x + alpha * y):02X}" for x, y in zip(a, b))


def add_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    size: float = 10,
    color: str = INK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.CENTER,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    font: str = FONT,
    margin: float = 0.02,
    name: str | None = None,
):
    box = slide.shapes.add_textbox(
        Inches(x * GEOM_SCALE),
        Inches(y * GEOM_SCALE),
        Inches(w * GEOM_SCALE),
        Inches(h * GEOM_SCALE),
    )
    if name:
        box.name = name
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin * GEOM_SCALE)
    tf.margin_right = Inches(margin * GEOM_SCALE)
    tf.margin_top = Inches(margin * GEOM_SCALE)
    tf.margin_bottom = Inches(margin * GEOM_SCALE)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size * TYPE_SCALE)
    r.font.bold = bold
    r.font.color.rgb = rgb(color)
    return box


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = WHITE,
    stroke: str = LINE,
    stroke_width: float = 0.9,
    radius: bool = True,
    name: str | None = None,
):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(
        kind,
        Inches(x * GEOM_SCALE),
        Inches(y * GEOM_SCALE),
        Inches(w * GEOM_SCALE),
        Inches(h * GEOM_SCALE),
    )
    if name:
        shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.color.rgb = rgb(stroke)
    shp.line.width = Pt(stroke_width)
    if radius:
        try:
            shp.adjustments[0] = 0.08
        except (IndexError, ValueError):
            pass
    return shp


def add_circle(
    slide,
    x: float,
    y: float,
    d: float,
    *,
    fill: str,
    stroke: str = WHITE,
    stroke_width: float = 0.7,
    name: str | None = None,
):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(x * GEOM_SCALE),
        Inches(y * GEOM_SCALE),
        Inches(d * GEOM_SCALE),
        Inches(d * GEOM_SCALE),
    )
    if name:
        shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.color.rgb = rgb(stroke)
    shp.line.width = Pt(stroke_width)
    return shp


def add_line(
    slide,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = LINE,
    width: float = 1.4,
    dashed: bool = False,
    arrow: bool = False,
    name: str | None = None,
):
    shp = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1 * GEOM_SCALE),
        Inches(y1 * GEOM_SCALE),
        Inches(x2 * GEOM_SCALE),
        Inches(y2 * GEOM_SCALE),
    )
    if name:
        shp.name = name
    shp.line.color.rgb = rgb(color)
    shp.line.width = Pt(width)
    if dashed:
        shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        ln = shp.line._get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "sm")
        tail.set("len", "sm")
        ln.append(tail)
    return shp


def add_section_tag(slide, x, y, w, text, color, name):
    add_text(
        slide,
        x,
        y,
        w,
        0.23,
        text,
        size=8.5,
        color=color,
        bold=True,
        align=PP_ALIGN.LEFT,
        name=name,
    )
    add_line(slide, x, y + 0.25, x + w, y + 0.25, color=color, width=1.2)


def add_panorama(
    slide,
    x,
    y,
    w,
    h,
    *,
    image_path: Path,
    marker_x_norm: float,
    marker_label: str,
    name: str,
):
    """Add a real 2:1 ERP and an editable declared-start marker."""
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    pic = slide.shapes.add_picture(
        str(image_path),
        Inches(x * GEOM_SCALE),
        Inches(y * GEOM_SCALE),
        Inches(w * GEOM_SCALE),
        Inches(h * GEOM_SCALE),
    )
    pic.name = name
    marker_x = x + marker_x_norm * w

    # Canonical equatorial start marker; retained after alignment.
    marker_y = y + 0.50 * h
    add_line(slide, marker_x, y + 0.05 * h, marker_x, marker_y - 0.03, color=CYAN, width=1.4)
    tri = slide.shapes.add_shape(
        MSO_SHAPE.ISOSCELES_TRIANGLE,
        Inches((marker_x - 0.055) * GEOM_SCALE),
        Inches((marker_y - 0.04) * GEOM_SCALE),
        Inches(0.11 * GEOM_SCALE),
        Inches(0.10 * GEOM_SCALE),
    )
    tri.name = f"{name}_start_marker"
    tri.rotation = 180
    tri.fill.solid()
    tri.fill.fore_color.rgb = rgb(CYAN)
    tri.line.color.rgb = rgb(CYAN)
    add_text(
        slide,
        marker_x - 0.17,
        y + 0.01,
        0.34,
        0.18,
        marker_label,
        size=7.5,
        color=CYAN,
        bold=True,
        font=MATH_FONT,
        name=f"{name}_g0_label",
    )
    frame = add_box(
        slide,
        x,
        y,
        w,
        h,
        fill=WHITE,
        stroke=LINE,
        stroke_width=0.85,
        radius=False,
        name=f"{name}_border",
    )
    frame.fill.background()
    return pic


def add_picture_panel(
    slide,
    image_path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    name: str,
    stroke: str = LINE,
    stroke_width: float = 0.7,
):
    """Embed a replaceable raster result while keeping an editable vector border."""
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    pic = slide.shapes.add_picture(
        str(image_path),
        Inches(x * GEOM_SCALE),
        Inches(y * GEOM_SCALE),
        Inches(w * GEOM_SCALE),
        Inches(h * GEOM_SCALE),
    )
    pic.name = name
    frame = add_box(
        slide,
        x,
        y,
        w,
        h,
        fill=WHITE,
        stroke=stroke,
        stroke_width=stroke_width,
        radius=False,
        name=f"{name}_border",
    )
    frame.fill.background()
    return pic


def mixture_value(x, y, centers):
    value = 0.0
    for cx, cy, sx, sy, amp in centers:
        dx = (x - cx) / sx
        dy = (y - cy) / sy
        value += amp * math.exp(-0.5 * (dx * dx + dy * dy))
    return value


def add_heatmap(
    slide,
    x,
    y,
    w,
    h,
    *,
    rows,
    cols,
    centers,
    low,
    high,
    name,
    border=LINE,
):
    values = []
    for r in range(rows):
        row = []
        for c in range(cols):
            px = (c + 0.5) / cols
            py = (r + 0.5) / rows
            row.append(mixture_value(px, py, centers))
        values.append(row)
    vmax = max(max(row) for row in values) or 1.0
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            v = max(0.0, min(1.0, values[r][c] / vmax))
            # Gamma makes low-density structure remain visible without fake peaks.
            col = blend(low, high, v ** 0.65)
            add_box(
                slide,
                x + c * cw,
                y + r * ch,
                cw + 0.001,
                ch + 0.001,
                fill=col,
                stroke=col,
                stroke_width=0,
                radius=False,
                name=f"{name}_cell_{r}_{c}",
            )
    add_box(
        slide,
        x,
        y,
        w,
        h,
        fill=WHITE,
        stroke=border,
        stroke_width=0.65,
        radius=False,
        name=f"{name}_border",
    ).fill.background()
    return values


def add_signed_residual(slide, x, y, w, h, name):
    rows, cols = 5, 10
    pos_centers = [(0.73, 0.43, 0.12, 0.22, 1.0), (0.32, 0.68, 0.14, 0.20, 0.65)]
    neg_centers = [(0.50, 0.48, 0.16, 0.24, 0.9), (0.88, 0.72, 0.10, 0.18, 0.45)]
    vals = []
    max_abs = 0.0
    for r in range(rows):
        row = []
        for c in range(cols):
            px, py = (c + 0.5) / cols, (r + 0.5) / rows
            v = mixture_value(px, py, pos_centers) - mixture_value(px, py, neg_centers)
            row.append(v)
            max_abs = max(max_abs, abs(v))
        vals.append(row)
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            v = vals[r][c] / (max_abs or 1.0)
            if v >= 0:
                col = blend("F8F8F6", ORANGE, v ** 0.7)
            else:
                col = blend("F8F8F6", TEAL, (-v) ** 0.7)
            add_box(
                slide,
                x + c * cw,
                y + r * ch,
                cw + 0.001,
                ch + 0.001,
                fill=col,
                stroke=col,
                stroke_width=0,
                radius=False,
                name=f"{name}_cell_{r}_{c}",
            )
    add_box(
        slide,
        x,
        y,
        w,
        h,
        fill=WHITE,
        stroke=TEAL,
        stroke_width=0.7,
        radius=False,
        name=f"{name}_border",
    ).fill.background()


def add_encoder_icon(slide, x, y, w, h):
    # Two branches stay compact because architecture is supporting evidence, not the claim.
    add_text(slide, x, y, w, 0.2, "DUAL-PATH ENCODER", size=7.2, color=MUTED, bold=True)
    y1 = y + 0.32
    y2 = y + 0.90
    cols = ["D6DDE1", "C4CED4", "AAB8C0"]
    for i, (bw, bh) in enumerate([(0.28, 0.44), (0.23, 0.35), (0.18, 0.26)]):
        add_box(
            slide,
            x + 0.08 + i * 0.34,
            y1 + (0.44 - bh) / 2,
            bw,
            bh,
            fill=cols[i],
            stroke="97A5AD",
            stroke_width=0.55,
            radius=False,
            name=f"Encoder_local_stage_{i + 1}",
        )
        if i < 2:
            add_line(
                slide,
                x + 0.08 + i * 0.34 + bw,
                y1 + 0.22,
                x + 0.08 + (i + 1) * 0.34,
                y1 + 0.22,
                color=LINE,
                width=0.8,
                arrow=True,
            )
    for i in range(3):
        add_box(
            slide,
            x + 0.08 + i * 0.34,
            y2,
            0.22,
            0.22,
            fill="E6E8EA",
            stroke="A6AFB5",
            stroke_width=0.5,
            radius=False,
            name=f"Encoder_semantic_stage_{i + 1}",
        )
        if i < 2:
            add_line(
                slide,
                x + 0.30 + i * 0.34,
                y2 + 0.11,
                x + 0.42 + i * 0.34,
                y2 + 0.11,
                color=LINE,
                width=0.8,
                arrow=True,
            )
    merge_x = x + w - 0.28
    add_line(slide, x + 1.0, y1 + 0.22, merge_x, y + 0.73, color=LINE, width=0.9, arrow=True)
    add_line(slide, x + 1.0, y2 + 0.11, merge_x, y + 0.73, color=LINE, width=0.9, arrow=True)
    add_circle(slide, merge_x, y + 0.61, 0.24, fill=PANEL_2, stroke=LINE, name="Encoder_fusion")
    add_text(slide, merge_x, y + 0.61, 0.24, 0.24, "+", size=10, color=MUTED, bold=True)
    add_text(
        slide,
        x,
        y + h - 0.26,
        w,
        0.24,
        "ERP-local  +  frozen semantic",
        size=7.2,
        color=MUTED,
    )


def add_gate_chart(slide, x, y, w, h, values):
    add_box(slide, x, y, w, h, fill=BLUE_LIGHT, stroke=BLUE, stroke_width=1.0, name="Monotonic_gate")
    add_text(
        slide,
        x + 0.08,
        y + 0.05,
        w - 0.16,
        0.30,
        "MONOTONIC GATE  βₜ",
        size=7.4,
        color=BLUE_DARK,
        bold=True,
        font=MATH_FONT,
    )
    px0, py0 = x + 0.18, y + h - 0.25
    px1, py1 = x + w - 0.12, y + 0.45
    add_line(slide, px0, py0, px1, py0, color=LINE, width=0.65)
    add_line(slide, px0, py0, px0, py1, color=LINE, width=0.65)
    # Keep the method diagram horizon-generic; concrete seconds belong in Experiments.
    ts = ["t₁", "t₂", "t₃", "t₄"]
    vals = [float(value) for value in values]
    if len(vals) != 4:
        raise ValueError(f"Expected four learned gates, got {len(vals)}")
    pts = []
    for i, val in enumerate(vals):
        px = px0 + (i + 0.12) / 3.25 * (px1 - px0)
        py = py0 - val * (py0 - py1)
        pts.append((px, py))
        if i:
            add_line(slide, pts[i - 1][0], pts[i - 1][1], px, py, color=BLUE, width=1.2)
        add_circle(slide, px - 0.035, py - 0.035, 0.07, fill=BLUE, stroke=WHITE, stroke_width=0.4)
        add_text(slide, px - 0.11, py0 + 0.01, 0.22, 0.16, ts[i], size=5.4, color=MUTED)


def add_formula_fusion(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=BLUE_LIGHT, stroke=BLUE, stroke_width=1.7, name="Core_innovation")
    add_box(
        slide,
        x + 0.43,
        y - 0.13,
        w - 0.86,
        0.30,
        fill=BLUE,
        stroke=BLUE,
        stroke_width=0,
        radius=True,
        name="Core_innovation_tag",
    )
    add_text(
        slide,
        x + 0.44,
        y - 0.13,
        w - 0.88,
        0.30,
        "CORE CONTRIBUTION",
        size=8.2,
        color=WHITE,
        bold=True,
    )
    add_text(
        slide,
        x + 0.12,
        y + 0.30,
        w - 0.24,
        0.38,
        "PRIOR-TO-EVIDENCE FACTORIZATION",
        size=9.0,
        color=BLUE_DARK,
        bold=True,
    )
    add_text(
        slide,
        x + 0.16,
        y + 0.80,
        w - 0.32,
        0.32,
        "population base measure",
        size=7.3,
        color=GOLD_DARK,
    )
    add_box(
        slide,
        x + 0.18,
        y + 1.12,
        w - 0.36,
        0.45,
        fill=GOLD_LIGHT,
        stroke=GOLD,
        stroke_width=0.8,
        radius=True,
        name="Prior_term",
    )
    add_text(
        slide,
        x + 0.18,
        y + 1.12,
        w - 0.36,
        0.45,
        "log Tₜ",
        size=15,
        color=GOLD_DARK,
        bold=True,
        font=MATH_FONT,
    )
    add_text(slide, x + 0.85, y + 1.61, 0.34, 0.34, "+", size=18, color=INK, bold=True, font=MATH_FONT)
    add_box(
        slide,
        x + 0.18,
        y + 1.98,
        w - 0.36,
        0.45,
        fill=TEAL_LIGHT,
        stroke=TEAL,
        stroke_width=0.8,
        radius=True,
        name="Evidence_term",
    )
    add_text(
        slide,
        x + 0.18,
        y + 1.98,
        w - 0.36,
        0.45,
        "βₜ Δₜ(Ĩ)",
        size=14,
        color=TEAL_DARK,
        bold=True,
        font=MATH_FONT,
    )
    add_text(
        slide,
        x + 0.12,
        y + h - 0.62,
        w - 0.24,
        0.43,
        "shared start structure\n+ image-specific re-ranking",
        size=7.1,
        color=MUTED,
    )


def add_map_and_tiles(slide, map_x, tile_x, y, map_w, tile_w, h, selected_mask):
    if not np.asarray(selected_mask).shape == (8, 16):
        raise ValueError(f"Expected an 8x16 selected-tile mask, got {np.asarray(selected_mask).shape}")
    add_text(slide, map_x, y - 0.48, map_w, 0.23, "EARLY MAP", size=7.5, color=INK, bold=True)
    add_text(slide, map_x, y - 0.25, map_w, 0.20, "Ĉₜ", size=10.5, color=BLUE_DARK, bold=True, font=MATH_FONT)
    add_picture_panel(
        slide,
        ACTUAL_DIR / "early_map_restored_t3.png",
        map_x,
        y,
        map_w,
        h,
        name="Predicted_early_map",
        stroke=INK,
        stroke_width=0.75,
    )
    add_text(slide, tile_x, y - 0.48, tile_w, 0.23, "TOP-B TILES", size=7.5, color=INK, bold=True)
    add_text(slide, tile_x, y - 0.25, tile_w, 0.20, "B / Q", size=7.1, color=MUTED, bold=True, font=MATH_FONT)

    rows, cols = 8, 16
    cw, ch = tile_w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            is_sel = bool(selected_mask[r, c])
            fill = CYAN if is_sel else WHITE
            stroke = CYAN if is_sel else "B5BEC4"
            add_box(
                slide,
                tile_x + c * cw,
                y + r * ch,
                cw,
                ch,
                fill=fill,
                stroke=stroke,
                stroke_width=0.38,
                radius=False,
                name=f"TopB_tile_{r}_{c}{'_selected' if is_sel else ''}",
            )
    add_box(
        slide,
        tile_x,
        y,
        tile_w,
        h,
        fill=WHITE,
        stroke=INK,
        stroke_width=0.75,
        radius=False,
        name="TopB_grid_border",
    ).fill.background()
    add_text(
        slide,
        tile_x - 0.08,
        y + h + 0.13,
        tile_w + 0.16,
        0.35,
        "same mass ranking\nfixed tile budget",
        size=6.6,
        color=MUTED,
        bold=True,
    )
    return int(np.asarray(selected_mask).sum())


def build_slide():
    if not ACTUALS_NPZ.is_file():
        raise FileNotFoundError(
            f"Missing real framework arrays: {ACTUALS_NPZ}. "
            "Run scripts/export_spherehorizon_framework_actuals.py first."
        )
    with np.load(ACTUALS_NPZ) as actuals:
        learned_gates = np.asarray(actuals["gates"], dtype=np.float32)
        selected_tile_mask = np.asarray(actuals["selected_tile_mask"], dtype=np.uint8)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W * GEOM_SCALE)
    prs.slide_height = Inches(SLIDE_H * GEOM_SCALE)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)

    # Lane containers first so all content remains selectable above them.
    add_box(slide, 4.42, 0.30, 3.18, 2.16, fill=TEAL_LIGHT, stroke=TEAL, stroke_width=1.15, name="Content_lane")
    add_box(slide, 4.42, 3.48, 3.18, 2.16, fill=GOLD_LIGHT, stroke=GOLD, stroke_width=1.15, name="Prior_lane")

    # Input and yaw-equivariant alignment.
    add_section_tag(slide, 0.22, 0.24, 3.75, "START-AWARE INPUT", INK, "Input_section_tag")
    add_panorama(
        slide,
        0.26,
        1.23,
        1.50,
        0.75,
        image_path=ERP_SOURCE,
        marker_x_norm=0.75,
        marker_label="g₀",
        name="Input_real_ERP_P10",
    )
    add_text(slide, 0.26, 2.12, 1.50, 0.27, "panorama I + declared start g₀", size=7.2, color=MUTED)

    add_circle(slide, 1.92, 1.39, 0.50, fill=PANEL, stroke=LINE, stroke_width=0.9, name="Yaw_alignment")
    add_text(slide, 1.92, 1.39, 0.50, 0.50, "R(g₀)", size=8.2, color=INK, bold=True, font=MATH_FONT)
    add_arrow = add_line
    add_arrow(slide, 1.77, 1.66, 1.91, 1.66, color=INK, width=1.15, arrow=True, name="Input_to_rotation")
    add_arrow(slide, 2.43, 1.66, 2.58, 1.66, color=INK, width=1.15, arrow=True, name="Rotation_to_aligned")

    add_panorama(
        slide,
        2.60,
        1.23,
        1.50,
        0.75,
        image_path=ERP_ALIGNED,
        marker_x_norm=0.50,
        marker_label="ref.",
        name="Aligned_real_ERP_P10_centered",
    )
    add_text(slide, 2.60, 2.12, 1.50, 0.27, "center-aligned ERP  Ĩ", size=7.2, color=MUTED, font=MATH_FONT)

    # Horizon node: dashed links are conditioning, not probability flow.
    add_circle(slide, 3.77, 2.68, 0.52, fill=BLUE_LIGHT, stroke=BLUE, stroke_width=1.0, name="Horizon_input")
    add_text(slide, 3.77, 2.68, 0.52, 0.52, "t", size=15, color=BLUE_DARK, bold=True, font=MATH_FONT)
    add_text(slide, 3.54, 3.19, 0.98, 0.23, "future horizon", size=6.8, color=MUTED)

    # Arrows are added before internal lane content when possible, so shapes mask crossings.
    add_line(slide, 4.10, 1.67, 4.56, 1.67, color=TEAL, width=1.4, arrow=True, name="Aligned_to_encoder")
    add_line(slide, 4.05, 2.68, 4.70, 1.05, color=BLUE, width=1.0, dashed=True, arrow=True, name="Time_conditions_content")
    add_line(slide, 4.04, 3.19, 4.73, 4.11, color=BLUE, width=1.0, dashed=True, arrow=True, name="Time_selects_template")
    add_line(slide, 4.29, 2.96, 7.72, 2.96, color=BLUE, width=1.0, dashed=True, arrow=True, name="Time_controls_gate")

    # Content evidence branch.
    add_text(slide, 4.57, 0.41, 2.88, 0.25, "CONTENT EVIDENCE", size=8.7, color=TEAL_DARK, bold=True, align=PP_ALIGN.LEFT)
    add_text(slide, 4.57, 0.65, 2.88, 0.23, "image-conditioned density-ratio correction", size=7.1, color=MUTED, align=PP_ALIGN.LEFT)
    add_encoder_icon(slide, 4.59, 0.92, 1.35, 1.35)
    add_line(slide, 5.96, 1.56, 6.18, 1.56, color=TEAL, width=1.25, arrow=True, name="Encoder_to_residual")
    add_picture_panel(
        slide,
        ACTUAL_DIR / "residual_t3.png",
        6.18,
        1.23,
        1.18,
        0.59,
        name="Actual_content_residual_P10_t3",
        stroke=TEAL,
        stroke_width=0.75,
    )
    add_text(slide, 6.16, 1.86, 1.22, 0.35, "actual log-ratio\nΔₜ(Ĩ)", size=6.7, color=TEAL_DARK, bold=True, font=MATH_FONT)

    # Population prior: empirical and asymmetric maps, not radial Gaussian cartoons.
    add_text(slide, 4.57, 3.58, 2.88, 0.25, "POPULATION PRIOR", size=8.7, color=GOLD_DARK, bold=True, align=PP_ALIGN.LEFT)
    add_text(slide, 4.57, 3.82, 2.88, 0.23, "start-aligned marginal density  Tₜ", size=7.1, color=MUTED, align=PP_ALIGN.LEFT, font=MATH_FONT)
    labels = ["t₁", "t₂", "t₃", "t₄"]
    positions = [(4.58, 4.15), (5.94, 4.15), (4.58, 4.80), (5.94, 4.80)]
    for i, (label, (hx, hy)) in enumerate(zip(labels, positions)):
        add_picture_panel(
            slide,
            ACTUAL_DIR / f"template_t{i + 1}.png",
            hx,
            hy,
            1.16,
            0.58,
            name=f"Actual_template_{i + 1}_P10",
            stroke=GOLD_DARK,
            stroke_width=0.55,
        )
        add_box(slide, hx + 0.03, hy + 0.03, 0.24, 0.17, fill=WHITE, stroke=GOLD, stroke_width=0.45, radius=True)
        add_text(slide, hx + 0.03, hy + 0.03, 0.24, 0.17, label, size=5.8, color=GOLD_DARK, bold=True)
    add_text(slide, 4.58, 5.42, 2.77, 0.18, "actual train-only templates", size=6.5, color=GOLD_DARK, bold=True)

    # Residual path and the separate horizon gate.
    add_line(slide, 7.35, 1.52, 8.02, 1.52, color=TEAL, width=1.55, arrow=True, name="Residual_to_gate_product")
    add_circle(slide, 8.02, 1.35, 0.34, fill=WHITE, stroke=TEAL, stroke_width=1.0, name="Gate_product")
    add_text(slide, 8.02, 1.35, 0.34, 0.34, "×", size=12, color=TEAL_DARK, bold=True)
    add_text(slide, 7.62, 1.06, 1.14, 0.22, "content correction", size=6.6, color=TEAL_DARK, bold=True)

    add_gate_chart(slide, 7.72, 2.58, 1.34, 1.30, learned_gates)
    add_line(slide, 8.39, 2.58, 8.19, 1.70, color=BLUE, width=1.05, dashed=True, arrow=True, name="Beta_modulates_residual")

    # Direct prior arrow: T_t does not generate beta_t.
    add_line(slide, 7.35, 4.55, 9.22, 3.95, color=GOLD, width=1.55, arrow=True, name="Log_template_to_fusion")
    add_text(slide, 7.62, 4.28, 1.10, 0.23, "log Tₜ", size=8.0, color=GOLD_DARK, bold=True, font=MATH_FONT)
    add_line(slide, 8.37, 1.52, 9.22, 1.86, color=TEAL, width=1.55, arrow=True, name="Gated_residual_to_fusion")
    add_text(slide, 8.35, 1.67, 0.78, 0.24, "βₜΔₜ", size=7.8, color=TEAL_DARK, bold=True, font=MATH_FONT)

    # Innovation core and standard post-processing.
    add_formula_fusion(slide, 9.27, 1.05, 2.18, 4.08)

    add_line(slide, 11.46, 3.02, 11.72, 3.02, color=INK, width=1.35, arrow=True, name="Fusion_to_softmax")
    add_box(slide, 11.74, 2.56, 0.82, 0.92, fill=PANEL, stroke=LINE, stroke_width=0.9, name="Spherical_softmax")
    add_text(slide, 11.76, 2.62, 0.78, 0.60, "spherical\nsoftmax", size=7.3, color=INK, bold=True)
    add_text(slide, 11.77, 3.20, 0.76, 0.18, "cos φ", size=6.2, color=MUTED, font=MATH_FONT)

    add_line(slide, 12.58, 3.02, 12.78, 3.02, color=INK, width=1.25, arrow=True, name="Softmax_to_inverse_rotation")
    add_circle(slide, 12.80, 2.70, 0.64, fill=PANEL, stroke=LINE, stroke_width=0.9, name="Inverse_yaw")
    add_text(slide, 12.80, 2.70, 0.64, 0.64, "R⁻¹(g₀)", size=8.2, color=INK, bold=True, font=MATH_FONT)
    add_text(slide, 12.71, 3.40, 0.82, 0.30, "restore yaw", size=6.4, color=MUTED)

    add_line(slide, 13.45, 3.02, 13.51, 3.02, color=INK, width=1.25, arrow=True, name="Inverse_rotation_to_map")
    add_map_and_tiles(slide, 13.53, 14.78, 2.70, 1.10, 1.10, 0.55, selected_tile_mask)
    add_line(slide, 14.65, 2.98, 14.76, 2.98, color=CYAN, width=1.25, arrow=True, name="Map_to_topB")

    # Compact semantics and line-style legend.
    add_text(
        slide,
        0.28,
        4.88,
        3.72,
        0.42,
        "History-free prediction\n(no fixation, observer ID, or online gaze)",
        size=8.2,
        color=INK,
        bold=True,
        align=PP_ALIGN.LEFT,
        valign=MSO_ANCHOR.TOP,
    )
    add_line(slide, 0.31, 5.56, 0.77, 5.56, color=INK, width=1.1, arrow=True)
    add_text(slide, 0.82, 5.43, 0.97, 0.25, "density / data", size=6.7, color=MUTED, align=PP_ALIGN.LEFT)
    add_line(slide, 1.92, 5.56, 2.38, 5.56, color=BLUE, width=1.0, dashed=True, arrow=True)
    add_text(slide, 2.43, 5.43, 1.28, 0.25, "horizon conditioning", size=6.7, color=MUTED, align=PP_ALIGN.LEFT)

    # Small application framing, subordinate to the method but closing the loop.
    add_text(
        slide,
        13.50,
        4.28,
        2.34,
        0.42,
        "APPLICATION LOOP",
        size=7.4,
        color=CYAN,
        bold=True,
    )
    add_text(
        slide,
        13.42,
        4.65,
        2.50,
        0.60,
        "rank spherical tile mass\n→ prefetch under a fixed budget",
        size=7.1,
        color=MUTED,
        bold=True,
    )

    # Remove the default first empty slide created by Presentation().
    if len(prs.slides) > 1:
        slide_id = prs.slides._sldIdLst[0]
        prs.part.drop_rel(slide_id.rId)
        del prs.slides._sldIdLst[0]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build_slide()
    print(path)
