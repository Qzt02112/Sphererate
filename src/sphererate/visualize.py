from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sphererate.utils import ensure_dir, read_jsonl


def _normalize_img(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    mx = float(np.nanmax(x))
    if mx <= 1e-7:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _heatmap(x: np.ndarray) -> Image.Image:
    x = _normalize_img(x)
    stops = np.array(
        [
            [20, 20, 80],
            [0, 145, 180],
            [80, 200, 120],
            [250, 220, 80],
            [220, 50, 35],
        ],
        dtype=np.float32,
    )
    pos = x * (len(stops) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _label(image: Image.Image, text: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return out


def _overlay_fixations(image: Image.Image, fixation: np.ndarray) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    ys, xs = np.nonzero(fixation > 0.5)
    for x, y in zip(xs[:: max(1, len(xs) // 800)], ys[:: max(1, len(ys) // 800)]):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 40, 40))
    return out


def make_gt_sheet(row: dict, out_path: str | Path) -> Path:
    image = Image.open(row["image"]).convert("RGB")
    target = np.load(row["target"])
    final = _heatmap(target["final"])
    fixation = _overlay_fixations(image, target["fixation"])
    temporal = [_heatmap(x) for x in target["temporal"]]
    cumulative = [_heatmap(x) for x in target["cumulative"]]

    tiles = [_label(image, "image"), _label(final, "final GT"), _label(fixation, "fixations")]
    tiles += [_label(tile, f"T{k+1}") for k, tile in enumerate(temporal)]
    tiles += [_label(tile, f"C{k+1}") for k, tile in enumerate(cumulative)]

    w, h = image.size
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), color=(24, 24, 24))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile.resize((w, h)), ((idx % cols) * w, (idx // cols) * h))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def make_gt_sheets(processed_root: str | Path, split: str, out_dir: str | Path, limit: int = 3) -> list[Path]:
    processed_root = Path(processed_root)
    out_dir = ensure_dir(out_dir)
    rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    paths = []
    for row in rows[:limit]:
        paths.append(make_gt_sheet(row, out_dir / f"{row['image_id']}_gt_sheet.png"))
    return paths
