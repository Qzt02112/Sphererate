from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sphererate.data.early_view import build_early_horizon_view, read_manifest


def _scanpath(path: Path, shift: float = 0.0) -> None:
    path.write_text(
        "Idx, longitude, latitude, start timestamp\n"
        f"0,{0.2 + shift},0.5,100\n"
        f"1,{0.3 + shift},0.5,900\n"
        "0,0.4,0.5,50\n"
        "1,0.5,0.5,1000\n",
        encoding="utf-8",
    )


def test_build_view_uses_right_eye_and_keeps_test_target_sealed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = []
    for index, split in enumerate(("train", "val", "gate", "test"), start=1):
        right = tmp_path / f"P{index}_R.txt"
        left = tmp_path / f"P{index}_L.txt"
        if split != "test":
            _scanpath(right)
            _scanpath(left, shift=0.1)
        rows.append(
            {
                "dataset": "salient360",
                "image_id": f"P{index}",
                "canonical_content_id": f"salient360:P{index}",
                "split": split,
                "image": str(tmp_path / f"P{index}.png"),
                "raw_scanpath_r": str(right),
                "raw_scanpath_l": str(left),
            }
        )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    root = build_early_horizon_view(
        source_manifest=source,
        out_root=tmp_path / "view",
        horizons_s=(0.5, 1.0),
        height=8,
        width=16,
        sigma_px=1.0,
        primary_eye="R",
        seal_test_targets=True,
    )
    manifest = read_manifest(root / "manifest.jsonl")

    assert len(manifest) == 4
    test_row = next(row for row in manifest if row["split"] == "test")
    assert test_row["target"] is None
    assert test_row["fixations"] is None
    assert test_row["target_sealed"] is True
    train_row = next(row for row in manifest if row["split"] == "train")
    with np.load(train_row["fixations"]) as data:
        assert len(np.unique(data["fixations"][:, 3])) == 2
    with np.load(train_row["target"]) as data:
        assert data["cumulative"].shape == (2, 8, 16)
        area = data["area_weights"]
        np.testing.assert_allclose((data["cumulative"] * area).sum(axis=(1, 2)), 1.0, atol=1e-5)
