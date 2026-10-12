from __future__ import annotations

import json

import numpy as np
import pytest

from sphererate.baselines.priors import make_prior_predictions


def _write_target(path, final: np.ndarray, area: np.ndarray, num_bins: int = 2) -> None:
    temporal = np.repeat(final[None], num_bins, axis=0)
    np.savez_compressed(
        path,
        final=final.astype(np.float32),
        temporal=temporal.astype(np.float32),
        cumulative=temporal.astype(np.float32),
        area_weights=area.astype(np.float32),
    )


@pytest.mark.parametrize("prediction_split", ["val", "test"])
def test_train_average_uses_only_train_targets_and_is_spherically_normalized(tmp_path, prediction_split: str) -> None:
    height, width = 4, 6
    area = np.linspace(0.25, 1.0, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    train_a = np.zeros((height, width), dtype=np.float32)
    train_b = np.zeros((height, width), dtype=np.float32)
    train_a[1, 1] = 2.0
    train_b[2, 4] = 3.0
    train_a /= float((train_a * area).sum())
    train_b /= float((train_b * area).sum())

    train_a_path = tmp_path / "train_a.npz"
    train_b_path = tmp_path / "train_b.npz"
    _write_target(train_a_path, train_a, area)
    _write_target(train_b_path, train_b, area)

    # This file intentionally does not exist. Success proves that prediction-target
    # GT is not loaded while constructing the empirical train prior.
    missing_prediction_target = tmp_path / "must_not_be_read.npz"
    prediction_id = f"{prediction_split}_a"
    rows = [
        {"image_id": "train_a", "split": "train", "target": str(train_a_path)},
        {"image_id": "train_b", "split": "train", "target": str(train_b_path)},
        {"image_id": prediction_id, "split": prediction_split, "target": str(missing_prediction_target)},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    out_dir = tmp_path / "predictions"
    make_prior_predictions(tmp_path, out_dir, kind="train_average", split=prediction_split)

    with np.load(out_dir / f"{prediction_id}.npz") as prediction:
        pred = prediction["final"]
        assert prediction["temporal"].shape == (2, height, width)
        assert prediction["cumulative"].shape == (2, height, width)
    expected = (train_a + train_b) / 2.0
    expected /= float((expected * area).sum())

    assert np.allclose(pred, expected, atol=1e-7)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    assert np.isclose((pred * area).sum(), 1.0, atol=1e-6)

    metadata = json.loads((out_dir / "prior_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_split"] == "train"
    assert metadata["source_count"] == 2
    assert metadata["prediction_split"] == prediction_split
    assert metadata["prediction_count"] == 1
