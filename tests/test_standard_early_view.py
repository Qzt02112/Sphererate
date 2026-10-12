from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.data.early_view import build_early_horizon_view, build_train_only_baselines, read_manifest


def _scanpath(path: Path, x: float) -> None:
    path.write_text(
        "Idx, longitude, latitude, start timestamp\n"
        f"0,{x},0.5,100\n1,{min(x + 0.03, 0.99)},0.5,900\n",
        encoding="utf-8",
    )


def test_standard_view_materializes_test_and_uses_train_only_template(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = []
    for index, split in enumerate(("train", "train", "val", "test"), start=1):
        image = tmp_path / f"P{index}.png"
        Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(image)
        raw = tmp_path / f"P{index}_R.txt"
        _scanpath(raw, 0.1 * index)
        rows.append(
            {
                "dataset": "salient360",
                "image_id": f"P{index}",
                "canonical_content_id": f"salient360:P{index}",
                "split": split,
                "image": str(image),
                "raw_scanpath_r": str(raw),
                "raw_scanpath_l": str(raw),
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    root = build_early_horizon_view(
        source_manifest=source,
        out_root=tmp_path / "view",
        horizons_s=(0.5, 1.0),
        height=8,
        width=16,
        sigma_px=1.0,
        seal_test_targets=False,
    )
    build_train_only_baselines(root, root / "baselines")
    built = read_manifest(root / "manifest.jsonl")

    assert all(row["target"] and Path(row["target"]).is_file() for row in built)
    assert all(row["fixations"] and Path(row["fixations"]).is_file() for row in built)
    assert all(not row["target_sealed"] for row in built)
    assert all((root / "baselines" / "early_template" / f"{row['image_id']}.npz").is_file() for row in built)

    with np.load(root / "baselines" / "early_template" / "P3.npz") as val_template:
        assert int(val_template["source_count"]) == 2
    with np.load(root / "baselines" / "early_template" / "P4.npz") as test_template:
        assert int(test_template["source_count"]) == 2
    with np.load(root / "baselines" / "early_template" / "P1.npz") as train_template:
        assert int(train_template["source_count"]) == 1

    sample = EarlyHorizonDataset(root, split="test")[0]
    assert sample["target"].shape == (2, 8, 16)
