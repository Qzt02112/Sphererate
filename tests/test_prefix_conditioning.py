from __future__ import annotations

import json

import numpy as np
from PIL import Image

from sphererate.data.conditioning import apply_prefix_condition_np
from sphererate.data.dataset import ProcessedSaliencyDataset
from sphererate.evaluation.process import evaluate_process_prediction_dir


def _write_prefix_fixture(root):
    height, width, num_bins = 4, 5, 4
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_path = root / "P1.png"
    Image.fromarray(image).save(image_path)

    area = np.ones((height, width), dtype=np.float32)
    temporal = np.zeros((num_bins, height, width), dtype=np.float32)
    for idx in range(num_bins):
        temporal[idx, idx % height, idx % width] = 1.0
    cumulative = temporal.cumsum(axis=0)
    target_path = root / "P1.npz"
    np.savez_compressed(
        target_path,
        temporal=temporal,
        cumulative=cumulative,
        final=temporal.sum(axis=0),
        fixation=(temporal.sum(axis=0) > 0).astype(np.float32),
        area_weights=area,
    )

    fixations = np.array(
        [
            [0.1, 0.1, 0.0, 0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 2.0, 0.0, 1.0, 2.0],
            [0.7, 0.7, 3.0, 0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    fixation_path = root / "P1_fix.npz"
    np.savez_compressed(fixation_path, fixations=fixations)

    manifest = {
        "image_id": "P1",
        "dataset": "demo",
        "split": "val",
        "image": str(image_path),
        "target": str(target_path),
        "fixations": str(fixation_path),
    }
    (root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return target_path


def test_prefix_conditioning_builds_future_target_and_channels(tmp_path) -> None:
    target_path = _write_prefix_fixture(tmp_path)
    with np.load(target_path) as data:
        target = {key: data[key].astype(np.float32) for key in data.files}

    conditioned, channels = apply_prefix_condition_np(target, prefix_bins=2, include_step_channel=True)

    assert channels.shape == (2, 4, 5)
    assert conditioned["temporal"].shape == (2, 4, 5)
    assert conditioned["cumulative"].shape == (2, 4, 5)
    assert conditioned["final"].shape == (4, 5)


def test_dataset_prefix_conditioning_extends_input_and_slices_targets(tmp_path) -> None:
    _write_prefix_fixture(tmp_path)
    dataset = ProcessedSaliencyDataset(
        tmp_path,
        split="val",
        condition_prefix_bins=2,
        condition_prefix_step_channel=True,
    )
    sample = dataset[0]

    assert sample["image"].shape == (10, 4, 5)
    assert sample["temporal"].shape == (2, 4, 5)
    assert sample["cumulative"].shape == (2, 4, 5)
    assert sample["final"].shape == (1, 4, 5)


def test_process_eval_uses_prefix_future_target(tmp_path) -> None:
    _write_prefix_fixture(tmp_path)
    preds = tmp_path / "preds"
    preds.mkdir()
    with np.load(tmp_path / "P1.npz") as data:
        target = {key: data[key].astype(np.float32) for key in data.files}
    conditioned, _ = apply_prefix_condition_np(target, prefix_bins=2, include_step_channel=True)
    np.savez_compressed(
        preds / "P1.npz",
        final=conditioned["final"],
        temporal=conditioned["temporal"],
        cumulative=conditioned["cumulative"],
    )

    summary = evaluate_process_prediction_dir(
        tmp_path,
        preds,
        split="val",
        condition_prefix_bins=2,
    )

    assert summary["condition_prefix_bins"] == 2
    assert summary["scores"]["temporal_kld_sph"] < 1e-5
