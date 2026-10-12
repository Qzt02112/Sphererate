from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
from PIL import Image


def tile_bounds(
    height: int, width: int, rows: int, cols: int
) -> list[tuple[int, int, int, int]]:
    """Return top/left/bottom/right bounds that exactly partition an ERP."""

    if min(height, width, rows, cols) <= 0 or rows > height or cols > width:
        raise ValueError("Invalid image or tile-grid dimensions")
    ys = np.linspace(0, height, rows + 1, dtype=np.int64)
    xs = np.linspace(0, width, cols + 1, dtype=np.int64)
    return [
        (int(ys[row]), int(xs[col]), int(ys[row + 1]), int(xs[col + 1]))
        for row in range(rows)
        for col in range(cols)
    ]


def encode_hevc_intra(
    tile_path: Path,
    bitstream_path: Path,
    qp: int,
    *,
    ffmpeg: str = "ffmpeg",
    threads: int = 1,
) -> None:
    """Encode one independently decodable HEVC-Intra tile with fixed QP."""

    if not 0 <= int(qp) <= 51:
        raise ValueError("HEVC QP must be in [0, 51]")
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    x265 = ":".join(
        (
            f"qp={int(qp)}",
            "keyint=1",
            "min-keyint=1",
            "scenecut=0",
            "repeat-headers=1",
            f"frame-threads={int(threads)}",
            "pools=none",
            "log-level=error",
        )
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(tile_path),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
        "-frames:v",
        "1",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-x265-params",
        x265,
        "-pix_fmt",
        "yuv420p",
        "-f",
        "hevc",
        str(bitstream_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HEVC encode failed: {result.stderr.strip()}")
    if not bitstream_path.is_file() or bitstream_path.stat().st_size == 0:
        raise RuntimeError(f"HEVC encoder produced no bitstream: {bitstream_path}")


def decode_hevc_intra(
    bitstream_path: Path,
    output_path: Path,
    *,
    crop_height: int | None = None,
    crop_width: int | None = None,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Decode one tile and optionally crop codec padding back to source size."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(bitstream_path),
        "-frames:v",
        "1",
    ]
    if crop_height is not None or crop_width is not None:
        if crop_height is None or crop_width is None:
            raise ValueError("crop_height and crop_width must be provided together")
        command.extend(
            ("-vf", f"crop={int(crop_width)}:{int(crop_height)}:0:0:exact=1")
        )
    command.append(str(output_path))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HEVC decode failed: {result.stderr.strip()}")


def allocate_upgrades(
    rank: np.ndarray,
    low_bytes: np.ndarray,
    high_bytes: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Greedily upgrade tiles in predicted order without exceeding total bytes."""

    order = np.asarray(rank, dtype=np.int64).reshape(-1)
    low = np.asarray(low_bytes, dtype=np.int64).reshape(-1)
    high = np.asarray(high_bytes, dtype=np.int64).reshape(-1)
    if low.shape != high.shape or len(order) != len(low) or set(order.tolist()) != set(range(len(low))):
        raise ValueError("rank must be a permutation aligned with byte arrays")
    if np.any(low < 0) or np.any(high < 0):
        raise ValueError("Expected non-negative byte counts")
    # A high-quality bitstream can occasionally be smaller for a very flat
    # tile.  Such Pareto-dominant upgrades are granted to every method before
    # ranking so the comparison never penalizes a method for codec noise.
    selected = high <= low
    current = int(low.sum() + (high[selected] - low[selected]).sum())
    if int(budget) < current:
        raise ValueError("Budget is smaller than the all-low representation")
    for tile_id in order:
        if selected[tile_id]:
            continue
        delta = int(high[tile_id] - low[tile_id])
        if current + delta <= int(budget):
            selected[tile_id] = True
            current += delta
    return selected


def reconstruct_tiles(
    low_tiles: Sequence[np.ndarray],
    high_tiles: Sequence[np.ndarray],
    upgraded: np.ndarray,
    *,
    height: int,
    width: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    if len(low_tiles) != rows * cols or len(high_tiles) != rows * cols:
        raise ValueError("Decoded tile count does not match grid")
    mask = np.asarray(upgraded, dtype=bool).reshape(-1)
    if len(mask) != rows * cols:
        raise ValueError("Upgrade mask does not match grid")
    sample = np.asarray(low_tiles[0])
    output = np.empty((height, width, *sample.shape[2:]), dtype=sample.dtype)
    for tile_id, (top, left, bottom, right) in enumerate(
        tile_bounds(height, width, rows, cols)
    ):
        tile = np.asarray(high_tiles[tile_id] if mask[tile_id] else low_tiles[tile_id])
        if tile.shape[:2] != (bottom - top, right - left):
            raise ValueError(f"Decoded tile {tile_id} has the wrong dimensions")
        output[top:bottom, left:right] = tile
    return output


def spherical_latitude_weights(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float64) + 0.5
    latitude = np.pi / 2.0 - np.pi * y / float(height)
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width)).copy()


def weighted_mse(
    reference: np.ndarray, reconstruction: np.ndarray, weights: np.ndarray
) -> float:
    reference_f = np.asarray(reference, dtype=np.float64)
    reconstruction_f = np.asarray(reconstruction, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if reference_f.shape != reconstruction_f.shape or reference_f.shape[:2] != weight.shape:
        raise ValueError("Reference, reconstruction, and weights are not aligned")
    pixel_error = np.mean((reference_f - reconstruction_f) ** 2, axis=-1)
    return float(np.sum(pixel_error * weight) / np.sum(weight).clip(min=1e-12))


def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(float(peak) ** 2 / mse)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
