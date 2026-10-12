#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.maps import build_temporal_maps
from sphererate.data.splits import audit_conditioned_pairs
from sphererate.utils import ensure_dir, read_jsonl, write_jsonl


JUFE_ID_RE = re.compile(r"^jufe_(?P<source>\d+_len\d+_(?:st|gn|gb|bd)_\d+)$")


def _load_startpoints(path: Path) -> dict[str, tuple[float, float]]:
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


def _raw_condition_files(raw_hm_root: Path, condition: str, source_id: str) -> list[Path]:
    return sorted((raw_hm_root / condition).glob(f"*/*/{source_id}.csv"))


def _point_to_norm(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    x_norm = ((float(lon_deg) + 180.0) / 360.0) % 1.0
    y_norm = np.clip((90.0 - float(lat_deg)) / 180.0, 0.0, np.nextafter(1.0, 0.0))
    return float(x_norm), float(y_norm)


def _load_condition_fixations(raw_hm_root: Path, condition: str, source_id: str, num_bins: int) -> np.ndarray:
    import xlrd

    rows: list[list[float]] = []
    files = _raw_condition_files(raw_hm_root, condition, source_id)
    if not files:
        raise FileNotFoundError(f"No JUFE HMData files found for condition={condition}, source={source_id}")

    for observer_id, path in enumerate(files):
        book = xlrd.open_workbook(str(path))
        sheet = book.sheet_by_index(0)
        steps = max(sheet.nrows, 1)
        for step_idx in range(sheet.nrows):
            values = sheet.row_values(step_idx)
            if len(values) < 5:
                continue
            valid = float(values[5]) if len(values) > 5 and values[5] != "" else 1.0
            if valid <= 0:
                continue
            lat = float(values[3]) if values[3] != "" else float(values[0])
            lon = float(values[4]) if values[4] != "" else float(values[1])
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            x_norm, y_norm = _point_to_norm(lon, lat)
            bin_idx = int(np.floor(step_idx * num_bins / steps))
            rows.append(
                [
                    x_norm,
                    y_norm,
                    float(step_idx),
                    float(observer_id),
                    1.0,
                    float(np.clip(bin_idx, 0, num_bins - 1)),
                ]
            )

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _safe_source_id(row: dict[str, Any]) -> str:
    source = str(row.get("source_image_id") or "")
    if source:
        return source
    match = JUFE_ID_RE.match(str(row["image_id"]))
    if not match:
        raise ValueError(f"Unexpected JUFE image_id: {row['image_id']}")
    return match.group("source")


def _copy_or_resize_image(src: Path, dst: Path, size_hw: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = size_hw
    with Image.open(src) as image:
        image = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        image.save(dst)


def _save_condition_target(
    path: Path,
    condition_maps: dict[str, np.ndarray],
    counterpart_maps: dict[str, np.ndarray],
    merged_maps: dict[str, np.ndarray],
    condition_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        temporal=condition_maps["temporal"],
        cumulative=condition_maps["cumulative"],
        final=condition_maps["final"],
        fixation=condition_maps["fixation"],
        area_weights=condition_maps["area_weights"],
        merged_temporal=merged_maps["temporal"],
        merged_cumulative=merged_maps["cumulative"],
        merged_final=merged_maps["final"],
        merged_fixation=merged_maps["fixation"],
        good_start_temporal=condition_maps["temporal"] if condition_name == "good" else counterpart_maps["temporal"],
        good_start_cumulative=condition_maps["cumulative"] if condition_name == "good" else counterpart_maps["cumulative"],
        good_start_final=condition_maps["final"] if condition_name == "good" else counterpart_maps["final"],
        good_start_fixation=condition_maps["fixation"] if condition_name == "good" else counterpart_maps["fixation"],
        bad_start_temporal=condition_maps["temporal"] if condition_name == "bad" else counterpart_maps["temporal"],
        bad_start_cumulative=condition_maps["cumulative"] if condition_name == "bad" else counterpart_maps["cumulative"],
        bad_start_final=condition_maps["final"] if condition_name == "bad" else counterpart_maps["final"],
        bad_start_fixation=condition_maps["fixation"] if condition_name == "bad" else counterpart_maps["fixation"],
    )


def _condition_manifest_row(
    row: dict[str, Any],
    source_id: str,
    sample_id: str,
    condition: str,
    counterpart: str,
    start_lon: float,
    other_lon: float,
    out_image: Path,
    target_path: Path,
    fixation_path: Path,
    raw_hm_root: Path,
) -> dict[str, Any]:
    return {
        "dataset": "jufe_start_conditioned",
        "image_id": sample_id,
        "source_image_id": source_id,
        "base_image_id": row["image_id"],
        "source_split": row.get("source_split"),
        "split": row["split"],
        "condition": condition,
        "counterpart_condition": counterpart,
        "start_lon_deg": float(start_lon),
        "start_lat_deg": 0.0,
        "counterpart_start_lon_deg": float(other_lon),
        "counterpart_start_lat_deg": 0.0,
        "image": str(out_image),
        "target": str(target_path),
        "fixations": str(fixation_path),
        "raw_image": row.get("raw_image"),
        "raw_hm_root": str(raw_hm_root),
        "source_processed_target": row.get("target"),
        "group_id": row.get("group_id"),
    }


def _reuse_conditioned_artifacts(
    source_root: Path,
    reuse_root: Path,
    out_root: Path,
    limit: int = 0,
) -> Path:
    """Create a clean manifest view over split-independent conditioned artifacts."""
    source_rows = [
        row for row in read_jsonl(source_root / "manifest.jsonl") if str(row.get("dataset")) == "jufe"
    ]
    if limit:
        source_rows = source_rows[: int(limit)]
    source_by_base = {str(row["image_id"]): row for row in source_rows}
    legacy_rows = read_jsonl(reuse_root / "manifest.jsonl")

    rows: list[dict[str, Any]] = []
    for legacy in legacy_rows:
        base_id = str(legacy.get("base_image_id") or f"jufe_{legacy.get('source_image_id')}")
        source = source_by_base.get(base_id)
        if source is None:
            continue
        row = dict(legacy)
        row["source_split"] = source.get("source_split")
        row["split"] = source["split"]
        row["group_id"] = source["group_id"]
        row["source_processed_target"] = source.get("target")
        row["clean_source_manifest"] = str(source_root / "manifest.jsonl")
        rows.append(row)

    expected_rows = len(source_rows) * 2
    if len(rows) != expected_rows:
        raise ValueError(
            f"Conditioned artifact coverage mismatch: expected {expected_rows} rows from "
            f"{len(source_rows)} JUFE sources, found {len(rows)}"
        )
    audit = audit_conditioned_pairs(rows)
    if not audit["zero_leakage"] or not audit["pair_complete"]:
        raise ValueError(
            f"Clean conditioned audit failed: leaks={audit['leaking_group_count']} "
            f"pair_complete={audit['pair_complete']}"
        )

    ensure_dir(out_root)
    write_jsonl(rows, out_root / "manifest.jsonl")
    with (out_root / "split_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    info = {
        "dataset": "jufe_start_conditioned",
        "source_processed": str(source_root),
        "artifact_source": str(reuse_root),
        "manifest_only_view": True,
        "num_source_images": len(source_rows),
        "num_conditioned_samples": len(rows),
        "image_size": [256, 512],
        "num_bins": 6,
        "sigma_px": 10.0,
        "area_weighted": True,
        "splits": {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]},
        "conditions": {
            condition: sum(row["condition"] == condition for row in rows) for condition in ["good", "bad"]
        },
        "split_audit": audit,
        "notes": (
            "Manifest-only clean group-disjoint view. Condition targets are split-independent artifacts "
            "built from raw JUFE HMData; split/group fields are inherited from clean Multi360 JUFE rows."
        ),
    }
    with (out_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return out_root


def prepare_jufe_start_conditioned(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_processed)
    out_root = Path(args.out_root)
    if args.reuse_conditioned_root:
        return _reuse_conditioned_artifacts(
            source_root,
            Path(args.reuse_conditioned_root),
            out_root,
            limit=int(args.limit),
        )
    raw_jufe_root = Path(args.raw_jufe_root)
    raw_hm_root = raw_jufe_root / "HMData_revised_final"
    startpoints = _load_startpoints(raw_jufe_root / "startpoint.xls")

    images_dir = ensure_dir(out_root / "images")
    targets_dir = ensure_dir(out_root / "targets")
    fixations_dir = ensure_dir(out_root / "fixations")

    source_rows = [
        row for row in read_jsonl(source_root / "manifest.jsonl") if str(row.get("dataset")) == "jufe"
    ]
    if args.limit:
        source_rows = source_rows[: int(args.limit)]

    h, w = int(args.height), int(args.width)
    rows: list[dict[str, Any]] = []
    for row in tqdm(source_rows, desc="Preparing JUFE start-conditioned"):
        source_id = _safe_source_id(row)
        image_filename = f"{source_id}.png"
        if image_filename not in startpoints:
            raise KeyError(f"Missing startpoint row for {image_filename}")
        good_start_lon, bad_start_lon = startpoints[image_filename]
        source_image = Path(row["image"])
        out_image = images_dir / f"jufe_{source_id}.png"
        if not out_image.exists() or args.overwrite:
            _copy_or_resize_image(source_image, out_image, (h, w))

        condition_specs: list[dict[str, Any]] = []
        for condition in ["good", "bad"]:
            start_lon = good_start_lon if condition == "good" else bad_start_lon
            other_lon = bad_start_lon if condition == "good" else good_start_lon
            sample_id = f"jufe_{source_id}__start_{condition}"
            target_path = targets_dir / f"{sample_id}.npz"
            fixation_path = fixations_dir / f"{sample_id}.npz"
            counterpart = "bad" if condition == "good" else "good"
            condition_specs.append(
                {
                    "condition": condition,
                    "counterpart": counterpart,
                    "start_lon": start_lon,
                    "other_lon": other_lon,
                    "sample_id": sample_id,
                    "target_path": target_path,
                    "fixation_path": fixation_path,
                }
            )

        if not args.overwrite and all(
            spec["target_path"].exists() and spec["fixation_path"].exists() for spec in condition_specs
        ):
            for spec in condition_specs:
                rows.append(
                    _condition_manifest_row(
                        row,
                        source_id,
                        spec["sample_id"],
                        spec["condition"],
                        spec["counterpart"],
                        spec["start_lon"],
                        spec["other_lon"],
                        out_image,
                        spec["target_path"],
                        spec["fixation_path"],
                        raw_hm_root,
                    )
                )
            continue

        fix_by_condition = {
            "good": _load_condition_fixations(raw_hm_root, "good", source_id, int(args.num_bins)),
            "bad": _load_condition_fixations(raw_hm_root, "bad", source_id, int(args.num_bins)),
        }
        maps_by_condition = {
            condition: build_temporal_maps(
                fixations,
                num_bins=int(args.num_bins),
                height=h,
                width=w,
                sigma_px=float(args.sigma_px),
                area_weighted=bool(args.area_weighted),
            )
            for condition, fixations in fix_by_condition.items()
        }
        merged_fixations = np.concatenate([fix_by_condition["good"], fix_by_condition["bad"]], axis=0)
        merged_maps = build_temporal_maps(
            merged_fixations,
            num_bins=int(args.num_bins),
            height=h,
            width=w,
            sigma_px=float(args.sigma_px),
            area_weighted=bool(args.area_weighted),
        )

        for spec in condition_specs:
            condition = spec["condition"]
            counterpart = spec["counterpart"]
            target_path = spec["target_path"]
            fixation_path = spec["fixation_path"]
            if args.overwrite or not (target_path.exists() and fixation_path.exists()):
                _save_condition_target(
                    target_path,
                    maps_by_condition[condition],
                    maps_by_condition[counterpart],
                    merged_maps,
                    condition,
                )
                np.savez_compressed(
                    fixation_path,
                    fixations=fix_by_condition[condition].astype(np.float32),
                    merged_fixations=merged_fixations.astype(np.float32),
                    columns=np.array(["x_norm", "y_norm", "timestamp", "observer_id", "duration", "bin"]),
                )

            rows.append(
                _condition_manifest_row(
                    row,
                    source_id,
                    spec["sample_id"],
                    condition,
                    counterpart,
                    spec["start_lon"],
                    spec["other_lon"],
                    out_image,
                    target_path,
                    fixation_path,
                    raw_hm_root,
                )
            )

    write_jsonl(rows, out_root / "manifest.jsonl")
    audit = audit_conditioned_pairs(rows)
    with (out_root / "split_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    info = {
        "dataset": "jufe_start_conditioned",
        "source_processed": str(source_root),
        "raw_jufe_root": str(raw_jufe_root),
        "num_source_images": len(source_rows),
        "num_conditioned_samples": len(rows),
        "image_size": [h, w],
        "num_bins": int(args.num_bins),
        "sigma_px": float(args.sigma_px),
        "area_weighted": bool(args.area_weighted),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]},
        "conditions": {condition: sum(row["condition"] == condition for row in rows) for condition in ["good", "bad"]},
        "target_keys": [
            "final",
            "temporal",
            "cumulative",
            "merged_final",
            "good_start_final",
            "bad_start_final",
        ],
        "notes": "Rows are start-conditioned good/bad samples from raw JUFE HMData. Raw files were read only.",
        "split_audit": audit,
    }
    with (out_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    readme_src = raw_jufe_root / "Readme.txt"
    if readme_src.exists():
        shutil.copyfile(readme_src, out_root / "JUFE_Readme.txt")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JUFE good/bad start-conditioned saliency targets.")
    parser.add_argument("--source-processed", default="data/processed/jufe_256x512_k6_order")
    parser.add_argument("--out-root", default="data/processed/jufe_start_conditioned_256x512_k6_order")
    parser.add_argument("--raw-jufe-root", default="${SCANDMM_ROOT}/data/raw/jufe")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=6)
    parser.add_argument("--sigma-px", type=float, default=10.0)
    parser.add_argument("--area-weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-conditioned-root",
        default=None,
        help="Reuse split-independent conditioned artifacts and rebuild only a clean manifest/audit.",
    )
    args = parser.parse_args()
    print(prepare_jufe_start_conditioned(args))


if __name__ == "__main__":
    main()
