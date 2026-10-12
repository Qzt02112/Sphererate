#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.data.maps import MapGeometry, erp_gaussian_map
from sphererate.visualize import _label


JUFE_ID_RE = re.compile(r"^jufe_(?P<content>\d+)_len(?P<length>\d+)_(?P<quality>st|gn|gb|bd)_(?P<start>\d+)$")
QUALITY_ORDER = {"st": 0, "gn": 1, "gb": 2, "bd": 3}


@dataclass(frozen=True)
class JufeRecord:
    image_id: str
    content: str
    length: str
    quality: str
    start: str
    split: str
    image: Path
    target: Path


def _read_manifest(path: Path) -> list[JufeRecord]:
    rows: list[JufeRecord] = []
    with path.open() as f:
        for line in f:
            raw = json.loads(line)
            match = JUFE_ID_RE.match(raw["image_id"])
            if not match:
                raise ValueError(f"Unexpected JUFE image_id: {raw['image_id']}")
            parts = match.groupdict()
            rows.append(
                JufeRecord(
                    image_id=raw["image_id"],
                    content=parts["content"],
                    length=parts["length"],
                    quality=parts["quality"],
                    start=parts["start"],
                    split=raw["split"],
                    image=Path(raw["image"]),
                    target=Path(raw["target"]),
                )
            )
    return rows


def _load_target(record: JufeRecord, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if record.image_id not in cache:
        data = np.load(record.target)
        cache[record.image_id] = (
            np.asarray(data["final"], dtype=np.float32),
            np.asarray(data["area_weights"], dtype=np.float32),
        )
    return cache[record.image_id]


def _sym_kld(a: np.ndarray, b: np.ndarray, area: np.ndarray) -> float:
    return 0.5 * (kld(a, b, area) + kld(b, a, area))


def _pair_metrics(a: JufeRecord, b: JufeRecord, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float | str]:
    amap, area = _load_target(a, cache)
    bmap, _ = _load_target(b, cache)
    return {
        "image_id_a": a.image_id,
        "image_id_b": b.image_id,
        "content_a": a.content,
        "content_b": b.content,
        "length_a": a.length,
        "length_b": b.length,
        "quality_a": a.quality,
        "quality_b": b.quality,
        "start_a": a.start,
        "start_b": b.start,
        "split_a": a.split,
        "split_b": b.split,
        "same_content": str(a.content == b.content),
        "same_length": str(a.length == b.length),
        "same_quality": str(a.quality == b.quality),
        "same_start": str(a.start == b.start),
        "cc_sph": cc(amap, bmap, area),
        "sim_sph": sim(amap, bmap, area),
        "sym_kld_sph": _sym_kld(amap, bmap, area),
    }


def _groups(records: list[JufeRecord], mode: str) -> dict[tuple[str, ...], list[JufeRecord]]:
    out: dict[tuple[str, ...], list[JufeRecord]] = defaultdict(list)
    for record in records:
        if mode == "content":
            key = (record.content,)
        elif mode == "content_len":
            key = (record.content, record.length)
        elif mode == "content_len_quality":
            key = (record.content, record.length, record.quality)
        else:
            raise ValueError(mode)
        out[key].append(record)
    return out


def _within_group_pairs(
    records: list[JufeRecord],
    mode: str,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group_key, group_records in _groups(records, mode).items():
        if len(group_records) < 2:
            continue
        for a, b in combinations(sorted(group_records, key=_record_sort_key), 2):
            row = _pair_metrics(a, b, cache)
            row["group_mode"] = mode
            row["group_key"] = "|".join(group_key)
            rows.append(row)
    return rows


def _random_cross_content_pairs(
    records: list[JufeRecord],
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
    num_pairs: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rng = random.Random(seed)
    rows: list[dict[str, float | str]] = []
    attempts = 0
    while len(rows) < num_pairs and attempts < num_pairs * 50:
        attempts += 1
        a, b = rng.sample(records, 2)
        if a.content == b.content:
            continue
        row = _pair_metrics(a, b, cache)
        row["group_mode"] = "random_cross_content"
        row["group_key"] = "random"
        rows.append(row)
    return rows


def _load_startpoint_table(path: Path) -> dict[str, tuple[float, float]]:
    import xlrd

    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    out: dict[str, tuple[float, float]] = {}
    for row_idx in range(1, sheet.nrows):
        values = sheet.row_values(row_idx)
        if len(values) < 3 or not values[0]:
            continue
        out[str(values[0])] = (float(values[1]), float(values[2]))
    return out


def _angular_delta_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _raw_condition_files(raw_hm_root: Path, condition: str, image_filename: str) -> list[Path]:
    return sorted((raw_hm_root / condition).glob(f"*/*/{image_filename}.csv"))


def _raw_condition_map(
    raw_hm_root: Path,
    condition: str,
    image_filename: str,
    height: int,
    width: int,
    sigma_px: float,
    area_weighted: bool,
    cache: dict[tuple[str, str], tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    key = (condition, image_filename)
    if key in cache:
        return cache[key]

    import xlrd

    xs: list[float] = []
    ys: list[float] = []
    files = _raw_condition_files(raw_hm_root, condition, image_filename)
    for path in files:
        book = xlrd.open_workbook(str(path))
        sheet = book.sheet_by_index(0)
        for row_idx in range(sheet.nrows):
            values = sheet.row_values(row_idx)
            if len(values) < 5:
                continue
            valid = float(values[5]) if len(values) > 5 and values[5] != "" else 1.0
            if valid <= 0:
                continue
            lat = float(values[3]) if values[3] != "" else float(values[0])
            lon = float(values[4]) if values[4] != "" else float(values[1])
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            xs.append(((lon + 180.0) / 360.0) % 1.0)
            ys.append(np.clip((90.0 - lat) / 180.0, 0.0, np.nextafter(1.0, 0.0)))

    area = MapGeometry(height, width).area_weights
    salmap = erp_gaussian_map(
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        height,
        width,
        sigma_px=sigma_px,
        area_weights=area,
        area_weighted=area_weighted,
    )
    cache[key] = (salmap.astype(np.float32), len(xs))
    return cache[key]


def _raw_good_bad_rows(
    records: list[JufeRecord],
    raw_jufe_root: Path,
    height: int,
    width: int,
    sigma_px: float,
    area_weighted: bool,
) -> tuple[list[dict[str, float | str]], dict[tuple[str, str], tuple[np.ndarray, int]]]:
    raw_hm_root = raw_jufe_root / "HMData_revised_final"
    startpoints = _load_startpoint_table(raw_jufe_root / "startpoint.xls")
    area = MapGeometry(height, width).area_weights
    cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
    rows: list[dict[str, float | str]] = []
    for idx, record in enumerate(records, start=1):
        image_filename = f"{record.content}_len{record.length}_{record.quality}_{record.start}.png"
        good_map, good_points = _raw_condition_map(
            raw_hm_root, "good", image_filename[:-4], height, width, sigma_px, area_weighted, cache
        )
        bad_map, bad_points = _raw_condition_map(
            raw_hm_root, "bad", image_filename[:-4], height, width, sigma_px, area_weighted, cache
        )
        good_start, bad_start = startpoints.get(image_filename, (float("nan"), float("nan")))
        rows.append(
            {
                "image_id": record.image_id,
                "image_filename": image_filename,
                "content": record.content,
                "length": record.length,
                "quality": record.quality,
                "start": record.start,
                "split": record.split,
                "good_points": good_points,
                "bad_points": bad_points,
                "good_start_value": good_start,
                "bad_start_value": bad_start,
                "start_delta_deg": _angular_delta_deg(good_start, bad_start)
                if np.isfinite(good_start) and np.isfinite(bad_start)
                else float("nan"),
                "cc_sph": cc(good_map, bad_map, area),
                "sim_sph": sim(good_map, bad_map, area),
                "sym_kld_sph": _sym_kld(good_map, bad_map, area),
            }
        )
        if idx % 100 == 0:
            print(f"processed raw good/bad maps: {idx}/{len(records)}", flush=True)
    return rows, cache


def _raw_cross_condition_random_rows(
    records: list[JufeRecord],
    raw_cache: dict[tuple[str, str], tuple[np.ndarray, int]],
    num_pairs: int,
    seed: int,
    height: int,
    width: int,
) -> list[dict[str, float | str]]:
    rng = random.Random(seed)
    area = MapGeometry(height, width).area_weights
    rows: list[dict[str, float | str]] = []
    attempts = 0
    while len(rows) < num_pairs and attempts < num_pairs * 50:
        attempts += 1
        a_record, b_record = rng.sample(records, 2)
        if a_record.image_id == b_record.image_id:
            continue
        a_filename = f"{a_record.content}_len{a_record.length}_{a_record.quality}_{a_record.start}.png"
        b_filename = f"{b_record.content}_len{b_record.length}_{b_record.quality}_{b_record.start}.png"
        good_map, _ = raw_cache[("good", a_filename[:-4])]
        bad_map, _ = raw_cache[("bad", b_filename[:-4])]
        rows.append(
            {
                "image_id_a": a_record.image_id,
                "image_id_b": b_record.image_id,
                "content_a": a_record.content,
                "content_b": b_record.content,
                "quality_a": a_record.quality,
                "quality_b": b_record.quality,
                "start_a": a_record.start,
                "start_b": b_record.start,
                "cc_sph": cc(good_map, bad_map, area),
                "sim_sph": sim(good_map, bad_map, area),
                "sym_kld_sph": _sym_kld(good_map, bad_map, area),
            }
        )
    return rows


def _record_sort_key(record: JufeRecord) -> tuple[int, int, int]:
    return (int(record.start), QUALITY_ORDER[record.quality], int(record.length))


def _float_values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _summary(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0}
    out: dict[str, float | int] = {"n": len(rows)}
    for key in ["sim_sph", "cc_sph", "sym_kld_sph"]:
        vals = sorted(_float_values(rows, key))
        out[f"{key}_mean"] = float(statistics.fmean(vals))
        out[f"{key}_median"] = float(statistics.median(vals))
        out[f"{key}_p10"] = float(np.percentile(vals, 10))
        out[f"{key}_p90"] = float(np.percentile(vals, 90))
    return out


def _summaries_by_relation(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        if row["group_mode"] == "random_cross_content":
            label = "random_cross_content"
        elif row["same_start"] == "True":
            label = f"{row['group_mode']}:same_start"
        else:
            label = f"{row['group_mode']}:different_start"
        buckets[label].append(row)
    for label in sorted(buckets):
        item = {"relation": label}
        item.update(_summary(buckets[label]))
        out.append(item)
    return out


def _start_pair_summaries(rows: list[dict[str, float | str]], group_mode: str) -> list[dict[str, float | int | str]]:
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        if row["group_mode"] != group_mode or row["same_start"] == "True":
            continue
        start_pair = "-".join(sorted([str(row["start_a"]), str(row["start_b"])]))
        buckets[start_pair].append(row)
    out: list[dict[str, float | int | str]] = []
    for label in sorted(buckets):
        item = {"start_pair": label, "group_mode": group_mode}
        item.update(_summary(buckets[label]))
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, float | int | str]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _heatmap_fixed(x: np.ndarray, vmax: float) -> Image.Image:
    x = np.asarray(x, dtype=np.float32)
    if vmax <= 1e-12:
        norm = np.zeros_like(x, dtype=np.float32)
    else:
        norm = np.clip(x / vmax, 0.0, 1.0)
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
    pos = norm * (len(stops) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _make_group_sheet(
    out_path: Path,
    records: list[JufeRecord],
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
    label: str,
    tile_width: int = 220,
) -> Path:
    records = sorted(records, key=_record_sort_key)
    maps = [_load_target(record, cache)[0] for record in records]
    vmax = max(float(np.max(x)) for x in maps)
    image_tiles: list[Image.Image] = []
    map_tiles: list[Image.Image] = []
    for record, salmap in zip(records, maps):
        image = Image.open(record.image).convert("RGB")
        w, h = image.size
        tile_h = int(round(tile_width * h / w))
        size = (tile_width, tile_h)
        tile_label = f"{record.image_id}  start={record.start}"
        image_tiles.append(_label(image.resize(size, Image.Resampling.BILINEAR), tile_label))
        map_tiles.append(_label(_heatmap_fixed(salmap, vmax).resize(size, Image.Resampling.BILINEAR), "GT final same scale"))

    sheet = Image.new("RGB", (tile_width * len(records), image_tiles[0].height * 2 + 24), color=(24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), label, fill=(255, 255, 255))
    y0 = 24
    for idx, tile in enumerate(image_tiles):
        sheet.paste(tile, (idx * tile_width, y0))
    for idx, tile in enumerate(map_tiles):
        sheet.paste(tile, (idx * tile_width, y0 + image_tiles[0].height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _make_raw_good_bad_sheet(
    out_path: Path,
    record: JufeRecord,
    raw_cache: dict[tuple[str, str], tuple[np.ndarray, int]],
    label: str,
    tile_width: int = 260,
) -> Path:
    image_filename = f"{record.content}_len{record.length}_{record.quality}_{record.start}.png"
    stem = image_filename[:-4]
    good_map, good_points = raw_cache[("good", stem)]
    bad_map, bad_points = raw_cache[("bad", stem)]
    vmax = max(float(np.max(good_map)), float(np.max(bad_map)))
    image = Image.open(record.image).convert("RGB")
    w, h = image.size
    tile_h = int(round(tile_width * h / w))
    size = (tile_width, tile_h)

    tiles = [
        _label(image.resize(size, Image.Resampling.BILINEAR), image_filename),
        _label(_heatmap_fixed(good_map, vmax).resize(size, Image.Resampling.BILINEAR), f"good start GT ({good_points} pts)"),
        _label(_heatmap_fixed(bad_map, vmax).resize(size, Image.Resampling.BILINEAR), f"bad start GT ({bad_points} pts)"),
    ]
    sheet = Image.new("RGB", (tile_width * len(tiles), tile_h + 24), color=(24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), label, fill=(255, 255, 255))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, (idx * tile_width, 24))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _content_group_scores(
    content_rows: list[dict[str, float | str]]
) -> list[dict[str, float | str]]:
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in content_rows:
        if row["same_start"] == "False":
            buckets[str(row["group_key"])].append(row)
    out: list[dict[str, float | str]] = []
    for group_key, rows in buckets.items():
        item = {"group_key": group_key}
        item.update(_summary(rows))
        out.append(item)
    return sorted(out, key=lambda row: float(row["sim_sph_mean"]))


def _select_example_groups(group_scores: list[dict[str, float | str]]) -> list[tuple[str, dict[str, float | str]]]:
    if not group_scores:
        return []
    picks = [
        ("low_similarity", group_scores[0]),
        ("median_similarity", group_scores[len(group_scores) // 2]),
        ("high_similarity", group_scores[-1]),
    ]
    unique: list[tuple[str, dict[str, float | str]]] = []
    seen: set[str] = set()
    for label, row in picks:
        key = str(row["group_key"])
        if key not in seen:
            unique.append((label, row))
            seen.add(key)
    return unique


def _select_example_rows(rows: list[dict[str, float | str]]) -> list[tuple[str, dict[str, float | str]]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: float(row["sim_sph"]))
    picks = [
        ("low_similarity", sorted_rows[0]),
        ("median_similarity", sorted_rows[len(sorted_rows) // 2]),
        ("high_similarity", sorted_rows[-1]),
    ]
    unique: list[tuple[str, dict[str, float | str]]] = []
    seen: set[str] = set()
    for label, row in picks:
        key = str(row["image_id"])
        if key not in seen:
            unique.append((label, row))
            seen.add(key)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JUFE GT salmap similarity across start IDs.")
    parser.add_argument("--manifest", default="data/processed/jufe_256x512_k6_order/manifest.jsonl")
    parser.add_argument("--out-dir", default="reports/jufe_start_salmap_similarity")
    parser.add_argument("--random-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--example-tile-width", type=int, default=220)
    parser.add_argument("--raw-jufe-root", default="${SCANDMM_ROOT}/data/raw/jufe")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--sigma-px", type=float, default=5.0)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_manifest(manifest)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    content_pairs = _within_group_pairs(records, "content", cache)
    content_len_pairs = _within_group_pairs(records, "content_len", cache)
    strict_pairs = _within_group_pairs(records, "content_len_quality", cache)
    cross_pairs = _random_cross_content_pairs(records, cache, args.random_pairs, args.seed)
    all_pairs = content_pairs + content_len_pairs + strict_pairs + cross_pairs

    relation_summaries = _summaries_by_relation(all_pairs)
    start_pair_summary = _start_pair_summaries(content_pairs, "content")
    content_group_summary = _content_group_scores(content_pairs)

    _write_csv(out_dir / "pairwise_within_content.csv", content_pairs)
    _write_csv(out_dir / "pairwise_within_content_len.csv", content_len_pairs)
    _write_csv(out_dir / "pairwise_strict_content_len_quality.csv", strict_pairs)
    _write_csv(out_dir / "pairwise_random_cross_content.csv", cross_pairs)
    _write_csv(out_dir / "summary_by_relation.csv", relation_summaries)
    _write_csv(out_dir / "summary_by_start_pair_content.csv", start_pair_summary)
    _write_csv(out_dir / "summary_by_content_group.csv", content_group_summary)

    by_content = _groups(records, "content")
    example_paths: list[Path] = []
    for label, row in _select_example_groups(content_group_summary):
        content = str(row["group_key"])
        group_records = by_content[(content,)]
        filename = f"{label}_content_{content}.png"
        path = _make_group_sheet(
            out_dir / "examples" / filename,
            group_records,
            cache,
            f"{label}: content={content}, diff-start mean SIM_sph={float(row['sim_sph_mean']):.3f}",
            tile_width=args.example_tile_width,
        )
        example_paths.append(path)

    content_group_sizes = Counter(len(v) for v in by_content.values())
    content_len_groups = _groups(records, "content_len")
    strict_groups = _groups(records, "content_len_quality")
    strict_multistart = sum(1 for v in strict_groups.values() if len({x.start for x in v}) > 1)
    content_len_multistart = sum(1 for v in content_len_groups.values() if len({x.start for x in v}) > 1)
    content_multistart = sum(1 for v in by_content.values() if len({x.start for x in v}) > 1)
    quality_start_counts = Counter((r.quality, r.start) for r in records)
    split_start_counts = Counter((r.split, r.start) for r in records)

    cache.clear()
    raw_rows: list[dict[str, float | str]] = []
    raw_random_rows: list[dict[str, float | str]] = []
    raw_summary_rows: list[dict[str, float | int | str]] = []
    raw_example_paths: list[Path] = []
    raw_jufe_root = Path(args.raw_jufe_root)
    if not args.skip_raw and (raw_jufe_root / "HMData_revised_final").exists():
        raw_rows, raw_cache = _raw_good_bad_rows(
            records=records,
            raw_jufe_root=raw_jufe_root,
            height=args.height,
            width=args.width,
            sigma_px=args.sigma_px,
            area_weighted=True,
        )
        raw_random_rows = _raw_cross_condition_random_rows(
            records=records,
            raw_cache=raw_cache,
            num_pairs=args.random_pairs,
            seed=args.seed,
            height=args.height,
            width=args.width,
        )
        raw_summary_rows = [
            {"relation": "raw_exact_same_image_good_vs_bad", **_summary(raw_rows)},
            {"relation": "raw_random_cross_image_good_vs_bad", **_summary(raw_random_rows)},
        ]
        _write_csv(out_dir / "pairwise_raw_exact_good_bad.csv", raw_rows)
        _write_csv(out_dir / "pairwise_raw_random_good_bad.csv", raw_random_rows)
        _write_csv(out_dir / "summary_raw_good_bad.csv", raw_summary_rows)
        record_by_id = {record.image_id: record for record in records}
        for label, row in _select_example_rows(raw_rows):
            image_id = str(row["image_id"])
            path = _make_raw_good_bad_sheet(
                out_dir / "examples" / f"raw_{label}_{image_id}.png",
                record_by_id[image_id],
                raw_cache,
                f"raw {label}: {image_id}, SIM_sph={float(row['sim_sph']):.3f}",
                tile_width=args.example_tile_width,
            )
            raw_example_paths.append(path)

    report = out_dir / "summary.md"
    report.write_text(
        "\n".join(
            [
                "# JUFE Start-Condition GT Salmap Similarity",
                "",
                "Date: 2026-07-10",
                "",
                "## Question",
                "",
                "Do different start IDs in JUFE produce similar final GT saliency maps?",
                "",
                "## Raw Good-vs-Bad Startpoint Analysis",
                "",
                "JUFE raw data contains separate `HMData_revised_final/good` and `HMData_revised_final/bad` gaze records for every distorted image. "
                "This is the strict same-image comparison we need for start-condition analysis. Each condition has 15 observer files per image; "
                "the script builds one final GT map per condition using the same ERP Gaussian map code as the main dataset pipeline.",
                "",
                _markdown_table(
                    raw_summary_rows,
                    [
                        "relation",
                        "n",
                        "sim_sph_mean",
                        "sim_sph_median",
                        "cc_sph_mean",
                        "cc_sph_median",
                        "sym_kld_sph_mean",
                        "sym_kld_sph_median",
                    ],
                )
                if raw_summary_rows
                else "Raw good/bad analysis was skipped or raw files were unavailable.",
                "",
                "Raw visual examples:",
                "",
                *[f"- [{path.name}]({path.relative_to(out_dir)})" for path in raw_example_paths],
                "",
                "## Processed Target Data Check",
                "",
                f"- Manifest: `{manifest}`",
                f"- Rows: `{len(records)}`",
                f"- Content IDs: `{len(by_content)}`",
                f"- Content group sizes: `{dict(sorted(content_group_sizes.items()))}`",
                f"- Strict same `content + len + quality` groups with multiple starts: `{strict_multistart}`",
                f"- Same `content + len` groups with multiple starts: `{content_len_multistart}` / `{len(content_len_groups)}`",
                f"- Same `content` groups with multiple starts: `{content_multistart}` / `{len(by_content)}`",
                "",
                "Important: the processed JUFE manifest/targets do **not** preserve separate good-start and bad-start final maps. "
                "Each exact distorted image (`content + len + quality`) has one merged target, so processed-level start-ID comparisons are exploratory.",
                "",
                "## Balance Check",
                "",
                "Quality by start:",
                "",
                _markdown_table(
                    [
                        {
                            "quality": quality,
                            "start1": quality_start_counts[(quality, "1")],
                            "start2": quality_start_counts[(quality, "2")],
                            "start3": quality_start_counts[(quality, "3")],
                        }
                        for quality in ["st", "gn", "gb", "bd"]
                    ],
                    ["quality", "start1", "start2", "start3"],
                ),
                "",
                "Split by start:",
                "",
                _markdown_table(
                    [
                        {
                            "split": split,
                            "start1": split_start_counts[(split, "1")],
                            "start2": split_start_counts[(split, "2")],
                            "start3": split_start_counts[(split, "3")],
                        }
                        for split in ["train", "val", "test"]
                    ],
                    ["split", "start1", "start2", "start3"],
                ),
                "",
                "## Processed Pairwise Similarity",
                "",
                _markdown_table(
                    relation_summaries,
                    [
                        "relation",
                        "n",
                        "sim_sph_mean",
                        "sim_sph_median",
                        "cc_sph_mean",
                        "cc_sph_median",
                        "sym_kld_sph_mean",
                        "sym_kld_sph_median",
                    ],
                ),
                "",
                "Interpretation: higher `SIM_sph` / `CC_sph` is more similar; lower symmetric `KLD_sph` is more similar.",
                "",
                "## Processed Start-Pair Breakdown Within Same Content",
                "",
                _markdown_table(
                    start_pair_summary,
                    [
                        "start_pair",
                        "n",
                        "sim_sph_mean",
                        "sim_sph_median",
                        "cc_sph_mean",
                        "cc_sph_median",
                        "sym_kld_sph_mean",
                        "sym_kld_sph_median",
                    ],
                ),
                "",
                "## Processed Visual Examples",
                "",
                *[f"- [{path.name}]({path.relative_to(out_dir)})" for path in example_paths],
                "",
                "## Takeaways",
                "",
                "1. Raw JUFE does contain strict same-image good-start vs bad-start gaze data, so this is the right level for start-condition analysis.",
                "2. Current processed JUFE targets are merged and cannot directly supervise or evaluate separate start-conditioned salmaps.",
                "3. Processed same-content different-start maps are more similar than random cross-content maps, so image/content remains a dominant driver.",
                "4. For final-map prediction under yaw-roll augmentation, predicting a fixed coordinate start is not a good mainline direction. "
                "A better use of JUFE is to model startpoint as a controlled condition or training-time auxiliary variable, and then marginalize/aggregate back to image-only final saliency.",
                "",
                "## Files",
                "",
                "- `pairwise_raw_exact_good_bad.csv`",
                "- `pairwise_raw_random_good_bad.csv`",
                "- `summary_raw_good_bad.csv`",
                "- `pairwise_within_content.csv`",
                "- `pairwise_within_content_len.csv`",
                "- `pairwise_strict_content_len_quality.csv`",
                "- `pairwise_random_cross_content.csv`",
                "- `summary_by_relation.csv`",
                "- `summary_by_start_pair_content.csv`",
                "- `summary_by_content_group.csv`",
            ]
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
