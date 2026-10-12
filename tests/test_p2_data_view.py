from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sphererate.data.p2_view import build_p2_temporal_aux_view


def _row(tmp_path: Path, image_id: str, split: str) -> dict:
    image_path = tmp_path / f"{image_id}.png"
    target_path = tmp_path / f"{image_id}.npz"
    fixation_path = tmp_path / f"{image_id}_fix.npz"
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(image_path)
    density = np.ones((2, 4), dtype=np.float32)
    np.savez_compressed(
        target_path,
        temporal=np.stack([density, density]),
        cumulative=np.stack([density, density]),
        final=density,
        fixation=density,
        area_weights=density,
    )
    np.savez_compressed(fixation_path, fixations=np.zeros((0, 6), dtype=np.float32))
    return {
        "dataset": "salient360",
        "image_id": image_id,
        "canonical_content_id": f"salient360:{image_id}",
        "group_id": f"content:salient360:{image_id}",
        "split": split,
        "image": str(image_path),
        "target": str(target_path),
        "fixations": str(fixation_path),
    }


def test_p2_view_makes_deterministic_nested_split_without_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = (
        [_row(tmp_path, f"train{i}", "train") for i in range(8)]
        + [_row(tmp_path, f"val{i}", "val") for i in range(2)]
        + [_row(tmp_path, f"test{i}", "test") for i in range(2)]
    )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    first = build_p2_temporal_aux_view(source, tmp_path / "out1", seed=11, dev_count=2)
    second = build_p2_temporal_aux_view(source, tmp_path / "out2", seed=11, dev_count=2)

    first_rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    second_rows = [json.loads(line) for line in second.read_text(encoding="utf-8").splitlines()]
    assert [(row["image_id"], row["split"]) for row in first_rows] == [
        (row["image_id"], row["split"]) for row in second_rows
    ]
    assert {split: sum(row["split"] == split for row in first_rows) for split in ("train", "val", "gate", "test")} == {
        "train": 6,
        "val": 2,
        "gate": 2,
        "test": 2,
    }
    assert all(row["outer_split"] == "train" for row in first_rows if row["split"] in {"train", "val"})
    assert all(row["outer_split"] == "val" for row in first_rows if row["split"] == "gate")
    groups = {
        split: {row["canonical_content_id"] for row in first_rows if row["split"] == split}
        for split in ("train", "val", "gate", "test")
    }
    for left_index, left in enumerate(groups):
        for right in list(groups)[left_index + 1 :]:
            assert groups[left].isdisjoint(groups[right])

    info = json.loads((tmp_path / "out1" / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["zero_leakage"] is True
    assert info["test_used"] is False

