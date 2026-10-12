from __future__ import annotations

import json

import numpy as np

from sphererate.evaluation.hard_subset import select_hard_rows, temporal_hardness_scores, write_hard_subset


def _write_target(path, first_xy: tuple[int, int], last_xy: tuple[int, int]) -> None:
    area = np.ones((4, 4), dtype=np.float32)
    temporal = np.zeros((2, 4, 4), dtype=np.float32)
    temporal[0, first_xy[1], first_xy[0]] = 1.0
    temporal[1, last_xy[1], last_xy[0]] = 1.0
    cumulative = np.stack([temporal[0], temporal[0] + temporal[1]], axis=0)
    np.savez(
        path,
        temporal=temporal,
        cumulative=cumulative,
        final=cumulative[-1],
        fixation=(cumulative[-1] > 0).astype(np.float32),
        area_weights=area,
    )


def test_temporal_hard_subset_ranks_moving_target_first(tmp_path):
    processed = tmp_path / "processed"
    targets = processed / "targets"
    targets.mkdir(parents=True)

    specs = {
        "P_easy": ((0, 0), (0, 0)),
        "P_mid": ((0, 0), (1, 0)),
        "P_hard": ((0, 0), (2, 3)),
    }
    manifest_rows = []
    for image_id, (first_xy, last_xy) in specs.items():
        target_path = targets / f"{image_id}.npz"
        _write_target(target_path, first_xy, last_xy)
        manifest_rows.append({"image_id": image_id, "split": "test", "target": str(target_path)})

    with open(processed / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    rows = temporal_hardness_scores(processed, split="test")
    selected = select_hard_rows(rows, top_k=1)
    write_hard_subset(rows, selected, csv_path=tmp_path / "hard.csv", id_path=tmp_path / "hard_ids.txt")

    assert rows[0]["image_id"] == "P_hard"
    assert selected[0]["image_id"] == "P_hard"
    assert "P_hard" in (tmp_path / "hard.csv").read_text(encoding="utf-8")
    assert (tmp_path / "hard_ids.txt").read_text(encoding="utf-8").strip() == "P_hard"
