from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sphererate.data.p2_view import build_standard_salient360_view


def _source_row(tmp_path: Path, image_id: str, historical_split: str) -> dict:
    image = tmp_path / f"{image_id}.png"
    target = tmp_path / f"{image_id}.npz"
    fixations = tmp_path / f"{image_id}_fix.npz"
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(image)
    np.savez_compressed(target, final=np.ones((2, 4), dtype=np.float32))
    np.savez_compressed(fixations, fixations=np.zeros((0, 6), dtype=np.float32))
    return {
        "dataset": "salient360",
        "image_id": image_id,
        "split": historical_split,
        "image": str(image),
        "target": str(target),
        "fixations": str(fixations),
    }


def test_standard_view_ignores_historical_manifest_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        _source_row(tmp_path, "P1", "test"),
        _source_row(tmp_path, "P2", "val"),
        _source_row(tmp_path, "P3", "train"),
        _source_row(tmp_path, "P4", "train"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 11,
                "group_to_split": {
                    "content:salient360:P1": "train",
                    "content:salient360:P2": "train",
                    "content:salient360:P3": "val",
                    "content:salient360:P4": "test",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_standard_salient360_view(source, split, tmp_path / "out")
    built = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assignments = {row["image_id"]: row["split"] for row in built}

    assert assignments == {"P1": "train", "P2": "train", "P3": "val", "P4": "test"}
    assert all(row["source_manifest_split"] != row["split"] for row in built)
    groups = {
        name: {row["canonical_content_id"] for row in built if row["split"] == name}
        for name in ("train", "val", "test")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    info = json.loads((tmp_path / "out" / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["counts"] == {"train": 2, "val": 1, "test": 1}
    assert info["zero_leakage"] is True
    assert "repeatable" in info["test_access_policy"]
