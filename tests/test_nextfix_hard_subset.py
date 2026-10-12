from __future__ import annotations

import json

import numpy as np
from PIL import Image

from sphererate.evaluation.nextfix_hard_subset import nextfix_hardness_scores, select_hard_rows, write_hard_subset


def test_nextfix_hard_subset_ranks_large_motion_sample_first(tmp_path) -> None:
    height, width = 4, 8
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_path = tmp_path / "img.png"
    Image.fromarray(image).save(image_path)
    fixations = np.array(
        [
            [0.10, 0.50, 0.0, 0.0, 1.0, 0.0],
            [0.12, 0.50, 1.0, 0.0, 1.0, 0.0],
            [0.14, 0.50, 2.0, 0.0, 1.0, 1.0],
            [0.16, 0.50, 3.0, 0.0, 1.0, 1.0],
            [0.48, 0.50, 4.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    fixation_path = tmp_path / "fix.npz"
    np.savez_compressed(fixation_path, fixations=fixations)
    manifest = {
        "dataset": "demo",
        "image_id": "img",
        "split": "train",
        "image": str(image_path),
        "fixations": str(fixation_path),
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    rows = nextfix_hardness_scores(tmp_path, split="train", prefix_length=3)
    selected = select_hard_rows(rows, top_k=1)
    write_hard_subset(rows, selected, csv_path=tmp_path / "hard.csv", sample_id_path=tmp_path / "ids.txt")

    assert rows[0]["sample_id"] == "img__obs0__t4"
    assert selected[0]["sample_id"] == "img__obs0__t4"
    assert "order_hardness_score" in rows[0]
    assert (tmp_path / "ids.txt").read_text(encoding="utf-8").strip() == "img__obs0__t4"
    assert "img__obs0__t4" in (tmp_path / "hard.csv").read_text(encoding="utf-8")
