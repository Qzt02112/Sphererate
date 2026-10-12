from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Salient360Item:
    image_id: str
    image_path: Path
    saliency_path: Path | None
    left_scanpath_path: Path | None
    right_scanpath_path: Path | None


def _numeric_id_from_name(name: str) -> str | None:
    match = re.search(r"(?:P|HEscanpath_|HEsalmap_)(\d+)", name)
    return match.group(1) if match else None


def discover_salient360(raw_root: str | Path) -> list[Salient360Item]:
    root = Path(raw_root) / "salient360"
    stimuli = root / "stimuli"
    salmaps = root / "SalMaps"
    scan_l = root / "Scanpaths" / "L"
    scan_r = root / "Scanpaths" / "R"

    image_by_id: dict[str, Path] = {}
    for path in stimuli.glob("*"):
        image_id = _numeric_id_from_name(path.name)
        if image_id and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_by_id[image_id] = path

    items: list[Salient360Item] = []
    for image_id, image_path in sorted(image_by_id.items(), key=lambda kv: int(kv[0])):
        sal_path = salmaps / f"HEsalmap_{image_id}.png"
        left_path = scan_l / f"HEscanpath_{image_id}.txt"
        right_path = scan_r / f"HEscanpath_{image_id}.txt"
        items.append(
            Salient360Item(
                image_id=image_id,
                image_path=image_path,
                saliency_path=sal_path if sal_path.exists() else None,
                left_scanpath_path=left_path if left_path.exists() else None,
                right_scanpath_path=right_path if right_path.exists() else None,
            )
        )
    return items


def _split_sequences(path: Path, eye: str, start_observer_id: int) -> list[np.ndarray]:
    sequences: list[list[tuple[float, float, float, str]]] = []
    current: list[tuple[float, float, float, str]] = []
    prev_idx: int | None = None
    with open(path, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(float(parts[0]))
                lon = float(parts[1])
                lat = float(parts[2])
                timestamp = float(parts[3])
            except ValueError:
                continue
            if prev_idx is not None and idx <= prev_idx and current:
                sequences.append(current)
                current = []
            current.append((lon, lat, timestamp, eye))
            prev_idx = idx
    if current:
        sequences.append(current)

    arrays: list[np.ndarray] = []
    for local_id, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        timestamps = np.array([row[2] for row in seq], dtype=np.float32)
        deltas = np.diff(timestamps, append=timestamps[-1])
        positive = deltas[deltas > 1e-3]
        fallback = float(np.median(positive)) if len(positive) else 1.0
        durations = np.where(deltas > 1e-3, deltas, fallback).astype(np.float32)
        arr = np.zeros((len(seq), 6), dtype=np.float32)
        for order, (lon, lat, timestamp, _eye) in enumerate(seq):
            arr[order, 0] = lon % 1.0
            arr[order, 1] = np.clip(lat, 0.0, 1.0)
            arr[order, 2] = timestamp
            arr[order, 3] = float(start_observer_id + local_id)
            arr[order, 4] = durations[order]
            arr[order, 5] = float(order)
        arrays.append(arr)
    return arrays


def load_salient360_fixations(item: Salient360Item, num_bins: int) -> np.ndarray:
    sequences: list[np.ndarray] = []
    observer_offset = 0
    if item.left_scanpath_path is not None:
        left = _split_sequences(item.left_scanpath_path, "L", observer_offset)
        sequences.extend(left)
        observer_offset += len(left)
    if item.right_scanpath_path is not None:
        right = _split_sequences(item.right_scanpath_path, "R", observer_offset)
        sequences.extend(right)

    rows: list[np.ndarray] = []
    for seq in sequences:
        n = len(seq)
        if n == 0:
            continue
        bins = np.floor(np.arange(n, dtype=np.float32) * num_bins / max(n, 1)).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        out = seq.copy()
        out[:, 5] = bins.astype(np.float32)
        rows.append(out)
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)
