from __future__ import annotations

import json

import numpy as np
from PIL import Image

from sphererate.data.dataset import ProcessedNextFixationDataset


def _write_scanpath_fixture(root):
    height, width = 6, 8
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_path = root / "img.png"
    Image.fromarray(image).save(image_path)

    fixations = np.array(
        [
            [0.10, 0.20, 0.0, 0.0, 1.0, 0.0],
            [0.30, 0.20, 1.0, 0.0, 1.0, 0.0],
            [0.50, 0.20, 2.0, 0.0, 1.0, 1.0],
            [0.70, 0.20, 3.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fixation_path = root / "fix.npz"
    np.savez_compressed(fixation_path, fixations=fixations)
    manifest = {
        "dataset": "demo",
        "image_id": "img",
        "split": "train",
        "image": str(image_path),
        "fixations": str(fixation_path),
    }
    (root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_next_fixation_dataset_builds_observer_prefix_samples(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    dataset = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="normal",
        sigma_px=0.5,
    )
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (11, 6, 8)
    assert sample["final"].shape == (1, 6, 8)
    assert sample["temporal"].shape == (1, 6, 8)
    assert str(sample["image_id"]).startswith("img__obs0__t2")
    assert sample["dataset"] == "demo"


def test_next_fixation_prefix_modes_change_current_fixation_channel(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    normal = ProcessedNextFixationDataset(tmp_path, split="train", prefix_length=2, prefix_mode="normal", sigma_px=0.5)[0]
    reversed_sample = ProcessedNextFixationDataset(tmp_path, split="train", prefix_length=2, prefix_mode="reversed", sigma_px=0.5)[0]
    none_sample = ProcessedNextFixationDataset(tmp_path, split="train", prefix_length=2, prefix_mode="none", sigma_px=0.5)[0]

    current_channel = 9
    prefix_channel = 8
    assert not np.allclose(normal["image"][current_channel].numpy(), reversed_sample["image"][current_channel].numpy())
    assert float(none_sample["image"][prefix_channel].sum()) == 0.0
    assert float(none_sample["image"][current_channel].sum()) == 0.0


def test_next_fixation_sequence_encoding_preserves_prefix_order(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    normal = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="normal",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    reversed_sample = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="reversed",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    none_sample = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="none",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]

    first_prefix_channel = 8
    second_prefix_channel = 9
    assert normal["image"].shape == (11, 6, 8)
    assert np.allclose(
        normal["image"][first_prefix_channel].numpy(),
        reversed_sample["image"][second_prefix_channel].numpy(),
    )
    assert np.allclose(
        normal["image"][second_prefix_channel].numpy(),
        reversed_sample["image"][first_prefix_channel].numpy(),
    )
    assert float(none_sample["image"][first_prefix_channel].sum()) == 0.0
    assert float(none_sample["image"][second_prefix_channel].sum()) == 0.0


def test_next_fixation_history_controls_keep_current_channel_fixed(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    normal = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="normal",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    history_reversed = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="history_reversed",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    current_only = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="current_only",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]

    first_prefix_channel = 8
    second_prefix_channel = 9
    current_channel = 10
    assert normal["image"].shape == (12, 6, 8)
    assert np.allclose(normal["image"][current_channel].numpy(), history_reversed["image"][current_channel].numpy())
    assert np.allclose(
        normal["image"][first_prefix_channel].numpy(),
        history_reversed["image"][second_prefix_channel].numpy(),
    )
    assert np.allclose(
        normal["image"][second_prefix_channel].numpy(),
        history_reversed["image"][first_prefix_channel].numpy(),
    )
    assert float(current_only["image"][first_prefix_channel].sum()) == 0.0
    assert float(current_only["image"][second_prefix_channel].sum()) == 0.0
    assert float(current_only["image"][current_channel].sum()) > 0.0


def test_next_fixation_start_only_uses_first_gt_fixation(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    normal = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="normal",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    start_only = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="start_only",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]
    start_repeat = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="start_repeat",
        prefix_encoding="sequence",
        sigma_px=0.5,
    )[0]

    first_prefix_channel = 8
    second_prefix_channel = 9
    current_channel = 10
    assert float(start_only["image"][first_prefix_channel].sum()) == 0.0
    assert float(start_only["image"][second_prefix_channel].sum()) == 0.0
    assert np.allclose(start_only["image"][current_channel].numpy(), normal["image"][first_prefix_channel].numpy())
    assert np.allclose(start_repeat["image"][first_prefix_channel].numpy(), normal["image"][first_prefix_channel].numpy())
    assert np.allclose(start_repeat["image"][second_prefix_channel].numpy(), normal["image"][first_prefix_channel].numpy())
    assert np.allclose(start_repeat["image"][current_channel].numpy(), normal["image"][first_prefix_channel].numpy())


def test_next_fixation_start_geometry_adds_relative_channels_only_for_start_modes(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    start_only = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="start_only",
        prefix_encoding="sequence",
        sigma_px=0.5,
        start_geometry=True,
    )[0]
    none_sample = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="none",
        prefix_encoding="sequence",
        sigma_px=0.5,
        start_geometry=True,
    )[0]

    start_geometry = start_only["image"][12:15].numpy()
    none_geometry = none_sample["image"][12:15].numpy()
    assert start_only["image"].shape == (15, 6, 8)
    assert np.isfinite(start_geometry).all()
    assert np.abs(start_geometry).max() <= 1.0
    assert float(np.abs(start_geometry).sum()) > 0.0
    assert float(np.abs(none_geometry).sum()) == 0.0


def test_next_fixation_negative_image_uses_same_target_with_perturbed_history(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    sample = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="normal",
        prefix_encoding="sequence",
        negative_prefix_mode="history_reversed",
        sigma_px=0.5,
    )[0]

    assert "negative_image" in sample
    assert sample["negative_image"].shape == sample["image"].shape
    assert np.allclose(sample["negative_image"][:8].numpy(), sample["image"][:8].numpy())
    assert np.allclose(sample["negative_image"][10].numpy(), sample["image"][10].numpy())
    assert not np.allclose(sample["negative_image"][8].numpy(), sample["image"][8].numpy())
    assert np.allclose(sample["negative_image"][11].numpy(), sample["image"][11].numpy())
    assert float(sample["final"].sum()) > 0.0


def test_next_fixation_multiple_negative_images_stack_modes(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    sample = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=3,
        prefix_mode="normal",
        prefix_encoding="sequence",
        negative_prefix_mode=["history_reversed", "current_only"],
        sigma_px=0.5,
    )[0]

    assert "negative_image" not in sample
    assert sample["negative_images"].shape == (2, *sample["image"].shape)
    assert np.allclose(sample["negative_images"][0, 10].numpy(), sample["image"][10].numpy())
    assert not np.allclose(sample["negative_images"][0, 8].numpy(), sample["image"][8].numpy())
    assert float(sample["negative_images"][1, 8].sum()) == 0.0
    assert float(sample["negative_images"][1, 9].sum()) == 0.0
    assert np.allclose(sample["negative_images"][1, 10].numpy(), sample["image"][10].numpy())


def test_next_fixation_limited_sampling_is_seeded_and_mode_invariant(tmp_path) -> None:
    height, width = 4, 6
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_path = tmp_path / "img.png"
    Image.fromarray(image).save(image_path)
    rows = []
    for image_idx in range(3):
        fixations = []
        for obs in range(2):
            for step in range(6):
                fixations.append([0.1 * (step + 1), 0.2, float(step), float(obs), 1.0, float(step // 2)])
        fixation_path = tmp_path / f"fix{image_idx}.npz"
        np.savez_compressed(fixation_path, fixations=np.asarray(fixations, dtype=np.float32))
        rows.append(
            {
                "dataset": "demo",
                "image_id": f"img{image_idx}",
                "split": "train",
                "image": str(image_path),
                "fixations": str(fixation_path),
            }
        )
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    normal = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="normal",
        max_samples=5,
        shuffle_seed=123,
    )
    shuffled = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="shuffled",
        max_samples=5,
        shuffle_seed=123,
    )
    other_seed = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        prefix_mode="normal",
        max_samples=5,
        shuffle_seed=456,
    )

    assert [row["sample_id"] for row in normal.rows] == [row["sample_id"] for row in shuffled.rows]
    assert [row["sample_id"] for row in normal.rows] != [row["sample_id"] for row in other_seed.rows]


def test_next_fixation_dataset_filters_sample_ids(tmp_path) -> None:
    _write_scanpath_fixture(tmp_path)
    sample_id_path = tmp_path / "sample_ids.txt"
    sample_id_path.write_text("img__obs0__t3\n", encoding="utf-8")

    dataset = ProcessedNextFixationDataset(
        tmp_path,
        split="train",
        prefix_length=2,
        sample_id_file=sample_id_path,
    )

    assert len(dataset) == 1
    assert dataset.rows[0]["sample_id"] == "img__obs0__t3"
