from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from sphererate.evaluation.codec_prefetch import (
    allocate_upgrades,
    decode_hevc_intra,
    encode_hevc_intra,
    psnr_from_mse,
    reconstruct_tiles,
    spherical_latitude_weights,
    tile_bounds,
    weighted_mse,
)


def test_tile_bounds_cover_each_pixel_once_for_odd_dimensions() -> None:
    height, width, rows, cols = 19, 35, 4, 6
    count = np.zeros((height, width), dtype=np.int64)
    for top, left, bottom, right in tile_bounds(height, width, rows, cols):
        count[top:bottom, left:right] += 1
    assert len(tile_bounds(height, width, rows, cols)) == rows * cols
    assert np.all(count == 1)


def test_allocate_upgrades_obeys_budget_and_skips_expensive_tile() -> None:
    low = np.array([10, 10, 10, 10])
    high = np.array([60, 15, 15, 15])
    selected = allocate_upgrades(np.array([0, 1, 2, 3]), low, high, budget=50)
    assert selected.tolist() == [False, True, True, False]
    actual = int(low.sum() + (high[selected] - low[selected]).sum())
    assert actual <= 50


def test_allocate_upgrades_grants_dominant_high_quality_tile_to_every_method() -> None:
    low = np.array([20, 10, 10])
    high = np.array([15, 20, 20])
    selected = allocate_upgrades(np.array([2, 1, 0]), low, high, budget=40)
    assert selected.tolist() == [True, False, False]


def test_reconstruct_tiles_matches_low_and_high_mosaics() -> None:
    height, width, rows, cols = 5, 7, 2, 2
    low, high = [], []
    for tile_id, (top, left, bottom, right) in enumerate(
        tile_bounds(height, width, rows, cols)
    ):
        shape = (bottom - top, right - left, 3)
        low.append(np.full(shape, tile_id, dtype=np.uint8))
        high.append(np.full(shape, 100 + tile_id, dtype=np.uint8))
    all_low = reconstruct_tiles(
        low, high, np.zeros(4, dtype=bool), height=height, width=width, rows=rows, cols=cols
    )
    all_high = reconstruct_tiles(
        low, high, np.ones(4, dtype=bool), height=height, width=width, rows=rows, cols=cols
    )
    assert np.max(all_low) == 3
    assert np.min(all_high) == 100


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_hevc_intra_is_repeatable_and_decodable_for_odd_tile(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    source = (rng.random((15, 23, 3)) * 255).astype(np.uint8)
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)
    decoded = []
    sizes = []
    for repeat in range(2):
        stream = tmp_path / f"encoded_{repeat}.hevc"
        output = tmp_path / f"decoded_{repeat}.png"
        encode_hevc_intra(source_path, stream, 32)
        decode_hevc_intra(stream, output, crop_height=15, crop_width=23)
        decoded.append(np.asarray(Image.open(output).convert("RGB")))
        sizes.append(stream.stat().st_size)
    assert sizes[0] == sizes[1]
    np.testing.assert_array_equal(decoded[0], decoded[1])
    assert decoded[0].shape == source.shape
    mse = weighted_mse(source, decoded[0], spherical_latitude_weights(15, 23))
    assert np.isfinite(psnr_from_mse(mse))
