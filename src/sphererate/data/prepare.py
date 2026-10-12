from __future__ import annotations

import csv
from collections import Counter
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from sphererate.data.maps import MapGeometry, build_temporal_maps, erp_gaussian_map
from sphererate.data.etdk2 import iter_viewing_sessions, normalize_session
from sphererate.data.salient360 import discover_salient360, load_salient360_fixations
from sphererate.data.splits import apply_group_disjoint_splits, audit_group_disjoint_splits
from sphererate.utils import ensure_dir, write_jsonl


SCANPATH_PICKLE_DATASETS = {"jufe", "sitzmann"}


def _load_etdk2_image_index(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {int(row["img_index"]): str(row["IMG"]).strip() for row in rows}
    if not mapping:
        raise ValueError(f"Empty ET-DK2 image index: {path}")
    return mapping


def _etdk2_session_fixations(trace: dict[str, np.ndarray], observer_id: int, num_bins: int) -> np.ndarray:
    valid = np.asarray(trace["gaze_valid"], dtype=bool)
    xy = np.asarray(trace["gaze_erp_xy"], dtype=np.float32)[valid]
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[valid]
    if len(xy) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if len(time_s) > 1:
        deltas = np.diff(time_s, append=time_s[-1])
        positive = deltas[deltas > 1e-6]
        fallback = float(np.median(positive)) if len(positive) else 1.0 / 60.0
        durations = np.where(deltas > 1e-6, deltas, fallback)
    else:
        durations = np.ones(1, dtype=np.float64) / 60.0
    total_duration = max(float(time_s[-1]), float(durations.sum()), 1e-7)
    bins = np.floor(time_s * float(num_bins) / total_duration).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)

    fixations = np.zeros((len(xy), 6), dtype=np.float32)
    fixations[:, :2] = xy
    fixations[:, 2] = time_s.astype(np.float32)
    fixations[:, 3] = float(observer_id)
    fixations[:, 4] = durations.astype(np.float32)
    fixations[:, 5] = bins.astype(np.float32)
    return fixations


def _resize_rgb(src: Path, dst: Path, size_hw: tuple[int, int]) -> None:
    h, w = size_hw
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        image.save(dst)


def _load_resized_gray(src: Path | None, size_hw: tuple[int, int]) -> np.ndarray | None:
    if src is None:
        return None
    h, w = size_hw
    with Image.open(src) as image:
        arr = np.asarray(image.convert("L").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = arr - arr.min()
    geom = MapGeometry(h, w)
    return erp_gaussian_map(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        h,
        w,
        sigma_px=1.0,
        area_weights=geom.area_weights,
    ) if arr.max() <= 0 else arr / (arr.sum() + 1e-7)


def _assign_splits(ids: list[str], val_fraction: float, test_fraction: float, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])
    return {
        image_id: "test" if image_id in test_ids else "val" if image_id in val_ids else "train"
        for image_id in ids
    }


def _normalize_split_id(image_id: str) -> str:
    text = str(image_id)
    return text[1:] if text.startswith("P") and text[1:].isdigit() else text


def _load_split_file(path: str | Path, ids: list[str]) -> dict[str, str]:
    split_path = Path(path)
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_by_id: dict[str, str] = {}
    valid = {"train", "val", "test"}
    if isinstance(data, dict) and any(key in data for key in valid):
        for split, image_ids in data.items():
            if split not in valid:
                continue
            for image_id in image_ids:
                split_by_id[_normalize_split_id(str(image_id))] = split
    elif isinstance(data, dict):
        for image_id, split in data.items():
            if split not in valid:
                raise ValueError(f"Invalid split '{split}' for image_id={image_id} in {split_path}")
            split_by_id[_normalize_split_id(str(image_id))] = str(split)
    else:
        raise ValueError(f"Split file must be a JSON object: {split_path}")

    missing = [image_id for image_id in ids if image_id not in split_by_id]
    if missing:
        preview = ", ".join(f"P{image_id}" for image_id in missing[:8])
        raise ValueError(f"Split file {split_path} is missing {len(missing)} ids: {preview}")
    return {image_id: split_by_id[image_id] for image_id in ids}


def _assign_train_val_from_predefined(
    train_ids: list[str],
    test_ids: list[str],
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = list(train_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if val_fraction > 0 and train_ids else 0
    val_ids = set(shuffled[:n_val])
    split_by_id = {image_id: "val" if image_id in val_ids else "train" for image_id in train_ids}
    split_by_id.update({image_id: "test" for image_id in test_ids})
    return split_by_id


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def _scanpath_processed_root(config: dict[str, Any]) -> Path:
    explicit = config["paths"].get("scandmm_processed_root")
    if explicit:
        return Path(explicit)
    raw_root = Path(config["paths"]["raw_root"])
    return raw_root.parent / "processed"


def _as_image_uint8(image: Any) -> np.ndarray:
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.min() < 0.0:
        arr = (arr + 1.0) * 0.5
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_resized_tensor_image(image: Any, dst: Path, size_hw: tuple[int, int]) -> None:
    h, w = size_hw
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_image_uint8(image)).resize((w, h), Image.Resampling.BILINEAR).save(dst)


def unit_vectors_to_fixations(scanpaths: np.ndarray, num_bins: int) -> np.ndarray:
    scanpaths = np.asarray(scanpaths, dtype=np.float32)
    if scanpaths.ndim == 2:
        scanpaths = scanpaths[None, ...]
    if scanpaths.ndim != 3 or scanpaths.shape[-1] != 3:
        raise ValueError(f"Expected scanpaths with shape [subjects, steps, 3], got {scanpaths.shape}")

    norm = np.linalg.norm(scanpaths, axis=-1, keepdims=True)
    valid = np.isfinite(scanpaths).all(axis=-1) & (norm.squeeze(-1) > 1e-8)
    unit = scanpaths / np.maximum(norm, 1e-8)
    lon = np.arctan2(unit[..., 1], unit[..., 0])
    lat = np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))
    x_norm = ((lon + np.pi) / (2.0 * np.pi)).astype(np.float32)
    y_norm = ((np.pi / 2.0 - lat) / np.pi).astype(np.float32)

    rows: list[list[float]] = []
    subjects, steps = scanpaths.shape[:2]
    for subject_idx in range(subjects):
        for step_idx in range(steps):
            if not bool(valid[subject_idx, step_idx]):
                continue
            bin_idx = int(np.floor(step_idx * num_bins / max(steps, 1)))
            rows.append(
                [
                    float(x_norm[subject_idx, step_idx] % 1.0),
                    float(np.clip(y_norm[subject_idx, step_idx], 0.0, 1.0)),
                    float(step_idx),
                    float(subject_idx),
                    1.0,
                    float(np.clip(bin_idx, 0, num_bins - 1)),
                ]
            )
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _scanpath_image_sources(dataset_name: str, image_id: str, config: dict[str, Any]) -> list[Path]:
    raw_root = Path(config["paths"]["raw_root"])
    scandmm_processed = _scanpath_processed_root(config)
    candidates: list[Path] = []
    if dataset_name == "jufe":
        candidates.append(raw_root / "jufe" / "dis" / f"{image_id}.png")
    candidates.append(scandmm_processed / dataset_name / "pathformer3d" / "images" / f"{image_id}.png")
    return [path for path in candidates if path.exists()]


def _resize_first_available_rgb(candidates: list[Path], dst: Path, size_hw: tuple[int, int]) -> Path | None:
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            _resize_rgb(candidate, dst, size_hw)
            return candidate
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None and not candidates:
        raise last_error
    return None


def _load_scanpath_pickle(dataset_name: str, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    pkl_path = data_cfg.get(f"{dataset_name}_pickle")
    if pkl_path is None:
        pkl_path = _scanpath_processed_root(config) / dataset_name / f"{dataset_name}.pkl"
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} pickle: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "train" not in data or "test" not in data:
        raise ValueError(f"Unsupported {dataset_name} pickle structure: {pkl_path}")
    return data


def prepare_etdk2(config: dict[str, Any]) -> Path:
    raw_root = Path(config["paths"]["raw_root"])
    processed_root = Path(config["paths"]["processed_root"])
    data_cfg = config["data"]
    h, w = tuple(data_cfg["image_size"])
    num_bins = int(data_cfg["num_bins"])
    out_root = processed_root / data_cfg["processed_name"]
    images_dir = ensure_dir(out_root / "images")
    targets_dir = ensure_dir(out_root / "targets")
    traces_dir = ensure_dir(out_root / "traces")
    fixations_dir = ensure_dir(out_root / "fixations")

    etdk2_root = Path(data_cfg.get("etdk2_raw_root") or raw_root / "etdk2")
    image_index = _load_etdk2_image_index(etdk2_root / "img_index.csv")
    participant_files = sorted((etdk2_root / "gaze").rglob("S*.csv"))
    if not participant_files:
        raise FileNotFoundError(f"No ET-DK2 participant CSV files below {etdk2_root / 'gaze'}")

    salient_items = {f"P{item.image_id}": item for item in discover_salient360(raw_root)}
    shared_stimuli = sorted(
        {stimulus for stimulus in image_index.values() if stimulus.startswith("P") and stimulus in salient_items},
        key=lambda name: int(name[1:]),
    )
    if not shared_stimuli:
        raise FileNotFoundError("No ET-DK2 P# stimuli match the local Salient360 images")
    split_by_stimulus = _assign_splits(
        shared_stimuli,
        float(data_cfg.get("val_fraction", 0.15)),
        float(data_cfg.get("test_fraction", 0.15)),
        int(data_cfg.get("split_seed", 20260711)),
    )

    min_samples = int(data_cfg.get("etdk2_min_session_samples", 100))
    min_duration_s = float(data_cfg.get("etdk2_min_session_duration_s", 24.0))
    force_rebuild = bool(data_cfg.get("etdk2_force_rebuild", False))
    excluded: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    resized_images: dict[str, Path] = {}

    for participant_path in tqdm(participant_files, desc="Preparing ET-DK2 subjects"):
        with participant_path.open("r", encoding="utf-8", newline="") as handle:
            sessions = iter_viewing_sessions(csv.DictReader(handle), image_index, shared_only=False)
            for session in sessions:
                trace = normalize_session(session.rows)
                duration_s = float(trace["time_s"][-1]) if len(trace["time_s"]) else 0.0
                if len(session.rows) < min_samples or duration_s < min_duration_s:
                    excluded["too_short_or_incomplete"] += 1
                    continue
                if not session.stimulus_id.startswith("P"):
                    excluded["construction_stimulus_unavailable"] += 1
                    continue
                item = salient_items.get(session.stimulus_id)
                if item is None:
                    excluded["missing_shared_salient360_image"] += 1
                    continue
                if not bool(np.asarray(trace["gaze_valid"], dtype=bool).any()):
                    excluded["no_valid_gaze"] += 1
                    continue

                if session.stimulus_id not in resized_images:
                    image_path = images_dir / f"{session.stimulus_id}.png"
                    if not image_path.is_file():
                        _resize_rgb(item.image_path, image_path, (h, w))
                    resized_images[session.stimulus_id] = image_path
                image_path = resized_images[session.stimulus_id]

                sample_id = f"etdk2_{session.stimulus_id}_{session.subject_id}_t{session.trial_index:02d}"
                trace_path = traces_dir / f"{sample_id}.npz"
                target_path = targets_dir / f"{sample_id}.npz"
                fixation_path = fixations_dir / f"{sample_id}.npz"
                if force_rebuild or not all(path.is_file() for path in (trace_path, target_path, fixation_path)):
                    np.savez_compressed(trace_path, **trace)
                    observer_id = int(session.subject_id[1:])
                    fixations = _etdk2_session_fixations(trace, observer_id=observer_id, num_bins=num_bins)
                    maps = build_temporal_maps(
                        fixations,
                        num_bins=num_bins,
                        height=h,
                        width=w,
                        sigma_px=float(data_cfg.get("sigma_px", 5.0)),
                        area_weighted=bool(data_cfg.get("area_weighted", True)),
                    )
                    np.savez_compressed(target_path, **maps)
                    np.savez_compressed(
                        fixation_path,
                        fixations=fixations.astype(np.float32),
                        columns=np.array(["x_norm", "y_norm", "timestamp", "observer_id", "duration", "bin"]),
                    )
                canonical_content_id = f"salient360:{session.stimulus_id}"
                rows.append(
                    {
                        "dataset": "etdk2",
                        "image_id": sample_id,
                        "source_image_id": session.stimulus_id,
                        "base_image_id": session.stimulus_id,
                        "canonical_content_id": canonical_content_id,
                        "group_id": f"content:{canonical_content_id}",
                        "subject_id": session.subject_id,
                        "trial_index": session.trial_index,
                        "start_index": session.start_index,
                        "start_longitude_deg": session.start_longitude_deg,
                        "start_longitude_declared_deg": session.start_longitude_deg,
                        "start_erp_longitude_deg": session.start_erp_longitude_deg,
                        "start_latitude_deg": session.start_latitude_deg,
                        "start_lon_deg": session.start_erp_longitude_deg,
                        "start_lat_deg": session.start_latitude_deg,
                        "duration_s": duration_s,
                        "num_samples": len(session.rows),
                        "num_valid_gaze_samples": int(np.asarray(trace["gaze_valid"], dtype=bool).sum()),
                        "num_valid_head_samples": int(np.asarray(trace["head_valid"], dtype=bool).sum()),
                        "raw_row_start": session.raw_row_start,
                        "raw_row_end": session.raw_row_end,
                        "split": split_by_stimulus[session.stimulus_id],
                        "image": str(image_path),
                        "trace": str(trace_path),
                        "target": str(target_path),
                        "fixations": str(fixation_path),
                        "raw_image": str(item.image_path),
                        "raw_gaze_csv": str(participant_path),
                    }
                )

    write_jsonl(rows, out_root / "manifest.jsonl")
    start_counts = Counter(str(row["start_index"]) for row in rows)
    subject_counts = Counter(str(row["subject_id"]) for row in rows)
    info = {
        "dataset": "etdk2",
        "variant": "salient360_shared_session_v1",
        "num_sessions": len(rows),
        "num_contents": len({row["canonical_content_id"] for row in rows}),
        "num_subjects": len(subject_counts),
        "image_size": [h, w],
        "num_bins": num_bins,
        "splits": {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]},
        "contents_by_split": {
            split: len({row["canonical_content_id"] for row in rows if row["split"] == split})
            for split in ["train", "val", "test"]
        },
        "sessions_by_start_index": dict(sorted(start_counts.items(), key=lambda item: int(item[0]))),
        "sessions_by_subject": dict(sorted(subject_counts.items())),
        "excluded_sessions": dict(sorted(excluded.items())),
        "raw_root": str(etdk2_root),
        "source_record": str(etdk2_root / "source_record.json") if (etdk2_root / "source_record.json").exists() else None,
        "license": "CC-BY-4.0",
        "coordinate_convention": {
            "texture_uv": "Unity bottom-left origin",
            "erp_array": "top-left origin with y_erp = 1 - y_texture",
            "start_longitude": "start_erp_longitude_deg = -start_longitude_declared_deg (wrapped)",
        },
        "notes": (
            "One row per valid subject-stimulus viewing session. Only the 40 public Salient360-linked P# stimuli "
            "are eligible; unavailable C# construction stimuli and incomplete one-row trials are excluded."
        ),
    }
    with (out_root / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    return out_root


def prepare_salient360(config: dict[str, Any]) -> Path:
    raw_root = Path(config["paths"]["raw_root"])
    processed_root = Path(config["paths"]["processed_root"])
    data_cfg = config["data"]
    h, w = tuple(data_cfg["image_size"])
    num_bins = int(data_cfg["num_bins"])
    out_root = processed_root / data_cfg["processed_name"]
    images_dir = ensure_dir(out_root / "images")
    targets_dir = ensure_dir(out_root / "targets")
    fix_dir = ensure_dir(out_root / "fixations")

    items = discover_salient360(raw_root)
    items = [item for item in items if item.left_scanpath_path or item.right_scanpath_path]
    max_images = data_cfg.get("max_images")
    if max_images:
        items = items[: int(max_images)]
    item_ids = [item.image_id for item in items]
    split_file = data_cfg.get("split_file")
    if split_file:
        split_by_id = _load_split_file(split_file, item_ids)
    else:
        split_by_id = _assign_splits(
            item_ids,
            float(data_cfg.get("val_fraction", 0.15)),
            float(data_cfg.get("test_fraction", 0.15)),
            int(data_cfg.get("split_seed", 20260708)),
        )

    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc="Preparing Salient360"):
        image_name = f"P{item.image_id}"
        resized_image = images_dir / f"{image_name}.png"
        target_path = targets_dir / f"{image_name}.npz"
        fixation_path = fix_dir / f"{image_name}.npz"

        _resize_rgb(item.image_path, resized_image, (h, w))
        fixations = load_salient360_fixations(item, num_bins=num_bins)
        maps = build_temporal_maps(
            fixations,
            num_bins=num_bins,
            height=h,
            width=w,
            sigma_px=float(data_cfg.get("sigma_px", 5.0)),
            area_weighted=bool(data_cfg.get("area_weighted", True)),
        )
        official = _load_resized_gray(item.saliency_path, (h, w))
        if official is not None:
            maps["official_final"] = official.astype(np.float32)

        np.savez_compressed(target_path, **maps)
        np.savez_compressed(
            fixation_path,
            fixations=fixations.astype(np.float32),
            columns=np.array(["x_norm", "y_norm", "timestamp", "observer_id", "duration", "bin"]),
        )
        rows.append(
            {
                "dataset": "salient360",
                "image_id": image_name,
                "split": split_by_id[item.image_id],
                "image": str(resized_image),
                "target": str(target_path),
                "fixations": str(fixation_path),
                "raw_image": str(item.image_path),
                "raw_saliency": str(item.saliency_path) if item.saliency_path else None,
                "raw_scanpath_l": str(item.left_scanpath_path) if item.left_scanpath_path else None,
                "raw_scanpath_r": str(item.right_scanpath_path) if item.right_scanpath_path else None,
            }
        )

    manifest_path = out_root / "manifest.jsonl"
    write_jsonl(rows, manifest_path)
    info = {
        "dataset": "salient360",
        "num_images": len(rows),
        "image_size": [h, w],
        "num_bins": num_bins,
        "splits": {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]},
        "split_file": str(split_file) if split_file else None,
        "raw_root": str(raw_root / "salient360"),
        "notes": "Generated from scanpaths only; raw files were not modified.",
    }
    with open(out_root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return out_root


def _scanpath_processed_name(dataset_name: str, data_cfg: dict[str, Any], h: int, w: int, num_bins: int) -> str:
    names = data_cfg.get("component_processed_names") or {}
    if dataset_name in names:
        return str(names[dataset_name])
    return f"{dataset_name}_{h}x{w}_k{num_bins}_order"


def prepare_scanpath_pickle_dataset(config: dict[str, Any], dataset_name: str) -> Path:
    if dataset_name not in SCANPATH_PICKLE_DATASETS:
        raise ValueError(f"Unsupported scanpath pickle dataset: {dataset_name}")

    processed_root = Path(config["paths"]["processed_root"])
    data_cfg = config["data"]
    h, w = tuple(data_cfg["image_size"])
    num_bins = int(data_cfg["num_bins"])
    out_root = processed_root / _scanpath_processed_name(dataset_name, data_cfg, h, w, num_bins)
    images_dir = ensure_dir(out_root / "images")
    targets_dir = ensure_dir(out_root / "targets")
    fix_dir = ensure_dir(out_root / "fixations")

    data = _load_scanpath_pickle(dataset_name, config)
    train_ids = sorted(str(image_id) for image_id in data["train"].keys())
    test_ids = sorted(str(image_id) for image_id in data["test"].keys())
    split_by_id = _assign_train_val_from_predefined(
        train_ids,
        test_ids,
        float(data_cfg.get("val_fraction", 0.15)),
        int(data_cfg.get("split_seed", 20260708)),
    )

    rows: list[dict[str, Any]] = []
    items = [(image_id, "train", data["train"][image_id]) for image_id in train_ids]
    items.extend((image_id, "test", data["test"][image_id]) for image_id in test_ids)
    for image_id, source_split, payload in tqdm(items, desc=f"Preparing {dataset_name}"):
        sample_id = f"{dataset_name}_{_safe_stem(image_id)}"
        image_path = images_dir / f"{sample_id}.png"
        target_path = targets_dir / f"{sample_id}.npz"
        fixation_path = fix_dir / f"{sample_id}.npz"

        src_image = None
        if not (image_path.exists() and target_path.exists() and fixation_path.exists()):
            src_image = _resize_first_available_rgb(_scanpath_image_sources(dataset_name, image_id, config), image_path, (h, w))
            if src_image is None:
                _save_resized_tensor_image(payload["image"], image_path, (h, w))

            fixations = unit_vectors_to_fixations(payload["scanpaths"], num_bins=num_bins)
            maps = build_temporal_maps(
                fixations,
                num_bins=num_bins,
                height=h,
                width=w,
                sigma_px=float(data_cfg.get("sigma_px", 5.0)),
                area_weighted=bool(data_cfg.get("area_weighted", True)),
            )
            np.savez_compressed(target_path, **maps)
            np.savez_compressed(
                fixation_path,
                fixations=fixations.astype(np.float32),
                columns=np.array(["x_norm", "y_norm", "timestamp", "observer_id", "duration", "bin"]),
            )
        else:
            sources = _scanpath_image_sources(dataset_name, image_id, config)
            src_image = sources[0] if sources else None
        rows.append(
            {
                "dataset": dataset_name,
                "image_id": sample_id,
                "source_image_id": image_id,
                "source_split": source_split,
                "split": split_by_id[image_id],
                "image": str(image_path),
                "target": str(target_path),
                "fixations": str(fixation_path),
                "raw_image": str(src_image) if src_image else None,
                "scanpath_pickle": str(data_cfg.get(f"{dataset_name}_pickle") or _scanpath_processed_root(config) / dataset_name / f"{dataset_name}.pkl"),
            }
        )

    write_jsonl(rows, out_root / "manifest.jsonl")
    info = {
        "dataset": dataset_name,
        "num_images": len(rows),
        "image_size": [h, w],
        "num_bins": num_bins,
        "splits": {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]},
        "source_splits": {split: sum(row["source_split"] == split for row in rows) for split in ["train", "test"]},
        "scanpath_pickle": str(data_cfg.get(f"{dataset_name}_pickle") or _scanpath_processed_root(config) / dataset_name / f"{dataset_name}.pkl"),
        "notes": "Generated from ScanDMM processed scanpath pickle; raw/source files were not modified.",
    }
    with open(out_root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return out_root


def _component_config(config: dict[str, Any], dataset_name: str, processed_name: str) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    out["data"]["dataset"] = dataset_name
    out["data"]["processed_name"] = processed_name
    if dataset_name != "salient360":
        out["data"]["split_file"] = None
    return out


def _read_component_rows(root: Path, dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(root / "manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_image_id = str(row["image_id"])
            if not source_image_id.startswith(f"{dataset_name}_"):
                row["source_image_id"] = source_image_id
                row["image_id"] = f"{dataset_name}_{_safe_stem(source_image_id)}"
            row["dataset"] = dataset_name
            rows.append(row)
    return rows


def prepare_multi360(config: dict[str, Any]) -> Path:
    processed_root = Path(config["paths"]["processed_root"])
    data_cfg = config["data"]
    h, w = tuple(data_cfg["image_size"])
    num_bins = int(data_cfg["num_bins"])
    dataset_names = [str(name).lower() for name in data_cfg.get("datasets", ["sitzmann", "salient360", "jufe"])]
    out_root = processed_root / data_cfg["processed_name"]
    ensure_dir(out_root)

    component_roots: dict[str, Path] = {}
    all_rows: list[dict[str, Any]] = []
    for name in dataset_names:
        if name == "salient360":
            processed_name = _scanpath_processed_name(name, data_cfg, h, w, num_bins)
            component_root = processed_root / processed_name
            if not (component_root / "manifest.jsonl").exists():
                prepare_salient360(_component_config(config, name, processed_name))
        elif name in SCANPATH_PICKLE_DATASETS:
            component_root = prepare_scanpath_pickle_dataset(config, name)
        else:
            raise ValueError(f"Unsupported multi360 component: {name}")

        component_roots[name] = component_root
        all_rows.extend(_read_component_rows(component_root, name))

    split_spec: dict[str, Any] | None = None
    if bool(data_cfg.get("group_disjoint", False)):
        all_rows, split_spec, split_audit = apply_group_disjoint_splits(
            all_rows,
            val_fraction=float(data_cfg.get("val_fraction", 0.15)),
            test_fraction=float(data_cfg.get("test_fraction", 0.15)),
            seed=int(data_cfg.get("split_seed", 20260710)),
        )
        if not split_audit["zero_leakage"]:
            raise RuntimeError(
                f"Group-disjoint split audit failed with {split_audit['leaking_group_count']} leaking groups."
            )
        split_manifest_path = Path(
            data_cfg.get("group_split_file") or out_root / "group_split_manifest.json"
        )
        split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with split_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(split_spec, f, indent=2, ensure_ascii=False)
    else:
        split_audit = audit_group_disjoint_splits(all_rows)
        split_manifest_path = None

    write_jsonl(all_rows, out_root / "manifest.jsonl")
    splits_by_dataset = {
        name: {split: sum(row["dataset"] == name and row["split"] == split for row in all_rows) for split in ["train", "val", "test"]}
        for name in dataset_names
    }
    info = {
        "dataset": "multi360",
        "datasets": dataset_names,
        "component_roots": {name: str(path) for name, path in component_roots.items()},
        "num_images": len(all_rows),
        "image_size": [h, w],
        "num_bins": num_bins,
        "splits": {split: sum(row["split"] == split for row in all_rows) for split in ["train", "val", "test"]},
        "splits_by_dataset": splits_by_dataset,
        "group_disjoint": bool(data_cfg.get("group_disjoint", False)),
        "group_split_file": str(split_manifest_path) if split_manifest_path else None,
        "split_audit": split_audit,
        "notes": (
            "Manifest-only group-disjoint dataset view; image and target artifacts remain in component roots."
            if bool(data_cfg.get("group_disjoint", False))
            else "Joint legacy manifest using component splits."
        ),
    }
    with open(out_root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return out_root


def prepare_dataset(config: dict[str, Any], dataset: str | None = None) -> Path:
    dataset_name = (dataset or config["data"]["dataset"]).lower()
    if dataset_name == "etdk2":
        return prepare_etdk2(config)
    if dataset_name == "salient360":
        return prepare_salient360(config)
    if dataset_name in SCANPATH_PICKLE_DATASETS:
        return prepare_scanpath_pickle_dataset(config, dataset_name)
    if dataset_name == "multi360":
        return prepare_multi360(config)
    raise ValueError(f"Unsupported dataset for preparation: {dataset_name}")
