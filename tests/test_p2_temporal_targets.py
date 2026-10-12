from __future__ import annotations

import json

import numpy as np
from PIL import Image
import torch

from sphererate.data.dataset import ProcessedSaliencyDataset


def _dataset_root(tmp_path) -> None:
    Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(tmp_path / "P1.png")
    temporal = np.stack([np.full((4, 8), value, dtype=np.float32) for value in (1, 2, 3)])
    final = np.full((4, 8), 7, dtype=np.float32)
    np.savez_compressed(
        tmp_path / "P1.npz",
        temporal=temporal,
        cumulative=np.cumsum(temporal, axis=0),
        final=final,
        fixation=np.ones((4, 8), dtype=np.float32),
        area_weights=np.ones((4, 8), dtype=np.float32),
    )
    row = {"image_id": "P1", "split": "train", "image": str(tmp_path / "P1.png"), "target": str(tmp_path / "P1.npz")}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_p2_target_modes_only_change_temporal_supervision(tmp_path) -> None:
    _dataset_root(tmp_path)
    modes = {
        mode: ProcessedSaliencyDataset(tmp_path, "train", temporal_target_mode=mode, shuffle_seed=7)[0]
        for mode in ("correct", "shuffled", "final_repeat")
    }
    assert torch.equal(modes["correct"]["image"], modes["shuffled"]["image"])
    assert torch.equal(modes["correct"]["final"], modes["final_repeat"]["final"])
    assert not torch.equal(modes["correct"]["temporal"], modes["shuffled"]["temporal"])
    assert torch.equal(modes["final_repeat"]["temporal"], modes["final_repeat"]["final"].repeat(3, 1, 1))
    assert torch.equal(modes["final_repeat"]["cumulative"], modes["final_repeat"]["temporal"])


def test_p2_target_mode_is_not_applied_outside_train(tmp_path) -> None:
    _dataset_root(tmp_path)
    row = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    row["split"] = "val"
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    correct = ProcessedSaliencyDataset(tmp_path, "val", temporal_target_mode="correct")[0]
    repeat = ProcessedSaliencyDataset(tmp_path, "val", temporal_target_mode="final_repeat")[0]
    assert torch.equal(correct["temporal"], repeat["temporal"])

