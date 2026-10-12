#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.utils import ensure_dir, read_jsonl
from sphererate.visualize import _heatmap, _label


DEFAULT_RUNS = {
    "Static": "runs/f2_256_static_30ep",
    "DyTSalDMM": "runs/f2_256_dytsaldmm_default_30ep",
    "DirectSum": "runs/f2_256_directsum_30ep",
    "TempSAL-style": "runs/f2_256_tempsal_mixer_30ep",
    "Residual": "runs/f2_256_residual_mixer_30ep",
    "Residual-noT": "runs/f2_256_residual_mixer_no_temporal_inputs_30ep",
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def _load_final_npz(path: Path, key: str = "final") -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} does not contain key '{key}'. Available keys: {data.files}")
    return np.asarray(data[key], dtype=np.float32)


def _resize_tile(tile: Image.Image, size: tuple[int, int]) -> Image.Image:
    return tile.resize(size, Image.Resampling.BILINEAR)


def _metric_lookup(report_root: Path, run_name: str, split: str) -> dict[str, dict[str, float]]:
    path = report_root / f"{run_name}_{split}" / f"{split}_per_image.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key == "image_id" or value in {"", None}:
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            out[row["image_id"]] = metrics
    return out


def _make_sheet(
    image_id: str,
    image_path: Path,
    target_path: Path,
    run_items: list[tuple[str, Path]],
    split: str,
    tile_width: int,
    pred_key: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    target = _load_final_npz(target_path, key="final")
    gt = _heatmap(target)

    w, h = image.size
    tile_h = int(round(tile_width * h / w))
    tile_size = (tile_width, tile_h)

    tiles = [
        _label(_resize_tile(image, tile_size), f"{image_id} image"),
        _label(_resize_tile(gt, tile_size), "GT final"),
    ]

    for label, run_dir in run_items:
        pred_path = run_dir / "predictions" / split / f"{image_id}.npz"
        if pred_path.exists():
            pred = _load_final_npz(pred_path, key=pred_key)
            tile = _heatmap(pred)
        else:
            tile = Image.new("RGB", image.size, color=(30, 30, 30))
            draw = ImageDraw.Draw(tile)
            draw.text((10, 10), "missing", fill=(240, 240, 240))
        tiles.append(_label(_resize_tile(tile, tile_size), label))

    sheet = Image.new("RGB", (len(tiles) * tile_width, tile_h), color=(24, 24, 24))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, (idx * tile_width, 0))
    return sheet


def _make_overview(rows: list[tuple[str, Image.Image]], out_path: Path, max_rows: int | None) -> Path:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("No rows to visualize.")
    width = max(sheet.width for _, sheet in rows)
    height = sum(sheet.height for _, sheet in rows)
    overview = Image.new("RGB", (width, height), color=(24, 24, 24))
    y = 0
    for _, sheet in rows:
        overview.paste(sheet, (0, y))
        y += sheet.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GT-vs-prediction saliency-map comparison sheets for trained runs."
    )
    parser.add_argument("--config", default="configs/salient360_256x512.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--out-dir", default="reports/f2_visualizations/core_models_test")
    parser.add_argument("--run", action="append", default=[], help="Run spec as Label=run_dir. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all common images in the split.")
    parser.add_argument("--overview-limit", type=int, default=0, help="0 means include all generated rows.")
    parser.add_argument("--tile-width", type=int, default=192)
    parser.add_argument("--pred-key", default="final")
    parser.add_argument(
        "--report-root",
        default=None,
        help="Optional root with per-image CSVs, used for sorting by first run KLD_sph.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    out_dir = ensure_dir(args.out_dir)

    run_items = [_parse_run(item) for item in args.run]
    if not run_items:
        run_items = [(label, Path(path)) for label, path in DEFAULT_RUNS.items()]

    rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == args.split]
    image_ids = [row["image_id"] for row in rows]

    if args.report_root:
        first_run_name = run_items[0][1].name
        metrics = _metric_lookup(Path(args.report_root), first_run_name, args.split)
        image_ids = sorted(image_ids, key=lambda image_id: metrics.get(image_id, {}).get("kld_sph", 1e9))

    if args.limit > 0:
        image_ids = image_ids[: args.limit]

    row_by_id = {row["image_id"]: row for row in rows}
    generated: list[tuple[str, Image.Image]] = []
    image_paths: list[Path] = []

    for image_id in image_ids:
        row = row_by_id[image_id]
        sheet = _make_sheet(
            image_id=image_id,
            image_path=Path(row["image"]),
            target_path=Path(row["target"]),
            run_items=run_items,
            split=args.split,
            tile_width=args.tile_width,
            pred_key=args.pred_key,
        )
        out_path = out_dir / f"{image_id}_{args.split}_gt_vs_predictions.png"
        sheet.save(out_path)
        generated.append((image_id, sheet))
        image_paths.append(out_path)

    overview_limit = None if args.overview_limit <= 0 else args.overview_limit
    overview_path = _make_overview(generated, out_dir / f"{args.split}_overview.png", overview_limit)

    index_path = out_dir / "index.md"
    with index_path.open("w") as f:
        f.write(f"# {args.split} GT vs Prediction Maps\n\n")
        f.write(f"- Processed data: `{processed_root}`\n")
        f.write(f"- Split: `{args.split}`\n")
        f.write(f"- Runs: {', '.join(label for label, _ in run_items)}\n")
        f.write(f"- Overview: `{overview_path.name}`\n\n")
        f.write(f"![overview]({overview_path.name})\n\n")
        f.write("## Per-Image Sheets\n\n")
        for path in image_paths:
            f.write(f"- [{path.name}]({path.name})\n")

    print(overview_path)
    print(index_path)
    for path in image_paths:
        print(path)


if __name__ == "__main__":
    main()
