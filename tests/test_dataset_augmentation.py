from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from sphererate.data.dataset import ProcessedSaliencyDataset, _startpoint_relative_channels, coord_channels


def test_yaw_roll_augments_maps_but_keeps_absolute_coord_channels(tmp_path) -> None:
    height, width = 4, 6
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    image_path = tmp_path / "P0.png"
    Image.fromarray(image).save(image_path)

    base_map = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    target_path = tmp_path / "P0.npz"
    np.savez_compressed(
        target_path,
        temporal=np.stack([base_map, base_map + 10], axis=0).astype(np.float32),
        cumulative=np.stack([base_map + 20, base_map + 30], axis=0).astype(np.float32),
        final=base_map.astype(np.float32),
        fixation=(base_map == 0).astype(np.float32),
        area_weights=np.ones((height, width), dtype=np.float32),
    )

    manifest = {
        "image_id": "P0",
        "split": "train",
        "image": str(image_path),
        "target": str(target_path),
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    dataset = ProcessedSaliencyDataset(
        tmp_path,
        split="train",
        augment_roll=True,
        augment_roll_degrees=[180],
    )
    sample = dataset[0]
    expected_shift = width // 2

    original_red = torch.from_numpy(image[..., 0].astype(np.float32) / 255.0)
    assert torch.allclose(sample["image"][0], torch.roll(original_red, shifts=expected_shift, dims=-1))

    assert torch.allclose(sample["image"][3:], coord_channels(height, width))
    assert torch.allclose(sample["final"][0], torch.roll(torch.from_numpy(base_map), shifts=expected_shift, dims=-1))
    assert torch.allclose(
        sample["temporal"][0],
        torch.roll(torch.from_numpy(base_map), shifts=expected_shift, dims=-1),
    )
    assert torch.allclose(
        sample["cumulative"][0],
        torch.roll(torch.from_numpy(base_map + 20), shifts=expected_shift, dims=-1),
    )
    assert torch.allclose(
        sample["fixation"][0],
        torch.roll(torch.from_numpy((base_map == 0).astype(np.float32)), shifts=expected_shift, dims=-1),
    )
    assert torch.allclose(sample["area_weights"][0], torch.ones((height, width), dtype=torch.float32))


def test_yaw_roll_keeps_start_relative_channels_aligned(tmp_path) -> None:
    height, width = 8, 16
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_path = tmp_path / "conditioned.png"
    Image.fromarray(image).save(image_path)
    target_path = tmp_path / "conditioned.npz"
    base_map = np.ones((height, width), dtype=np.float32)
    area = np.ones((height, width), dtype=np.float32)
    np.savez_compressed(
        target_path,
        temporal=np.stack([base_map, base_map]),
        cumulative=np.stack([base_map, base_map]),
        final=base_map,
        fixation=base_map,
        area_weights=area,
    )
    row = {
        "image_id": "conditioned",
        "split": "train",
        "image": str(image_path),
        "target": str(target_path),
        "start_lon_deg": 30.0,
        "start_lat_deg": 0.0,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = ProcessedSaliencyDataset(
        tmp_path,
        split="train",
        augment_roll=True,
        augment_roll_degrees=[180],
        condition_startpoint_channels=True,
        startpoint_prior_sigma_px=1.0,
    )
    sample = dataset[0]
    rotated_lon = -150.0
    expected = _startpoint_relative_channels(
        {"image_id": "conditioned", "start_lon_deg": rotated_lon, "start_lat_deg": 0.0},
        height,
        width,
        area,
        sigma_px=1.0,
    )
    assert torch.allclose(sample["image"][8:13], torch.from_numpy(expected), atol=1e-6)
