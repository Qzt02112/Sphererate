from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sphererate.data.dataset import coord_channels
from sphererate.data.early_horizon import raw_dwell_tile_distribution
from sphererate.data.early_view import read_manifest


def _derangement(length: int, key: str, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    identity = np.arange(length)
    for _ in range(100):
        permutation = rng.permutation(length)
        if np.all(permutation != identity):
            return permutation
    return np.roll(identity, 1)


def _declared_start_geometry(row: dict, height: int, width: int) -> np.ndarray:
    """Return spherical distance/cosine channels from the declared launch point.

    The early-attention task exposes the viewing start as an input contract.  We
    keep it separate from RGB/coordinate channels so checkpoints of the matched
    semantic encoder remain reusable and start information cannot be mistaken
    for image content.
    """

    # Coordinate-only cross-dataset views explicitly record the start in the
    # model source's longitude convention.  Prefer that value over the source
    # manifest's historical pixel-angle field (notably Sitzmann's 180-degree
    # ERP-centre convention) so explicit spherical geometry follows the rolled
    # RGB/targets rather than silently returning to the old coordinate system.
    yaw = np.deg2rad(
        float(
            row.get(
                "model_declared_start_yaw_deg",
                row.get(
                    "external_scanpath_start_yaw_deg",
                    row.get("declared_start_yaw_deg", 0.0),
                ),
            )
        )
    )
    latitude = np.deg2rad(float(row.get("declared_start_lat_deg", 0.0)))
    x = (np.arange(int(width), dtype=np.float64) + 0.5) / float(width)
    y = (np.arange(int(height), dtype=np.float64) + 0.5) / float(height)
    longitude_grid, latitude_grid = np.meshgrid(
        2.0 * np.pi * x - np.pi,
        np.pi / 2.0 - np.pi * y,
    )
    cosine = (
        np.sin(latitude_grid) * np.sin(latitude)
        + np.cos(latitude_grid) * np.cos(latitude) * np.cos(longitude_grid - yaw)
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    distance = np.arccos(cosine)
    east = np.cos(latitude_grid) * np.sin(longitude_grid - yaw)
    north = (
        np.sin(latitude_grid) * np.cos(latitude)
        - np.cos(latitude_grid) * np.sin(latitude) * np.cos(longitude_grid - yaw)
    )
    sine_distance = np.sin(distance)
    log_scale = np.divide(
        distance,
        sine_distance,
        out=np.ones_like(distance),
        where=np.abs(sine_distance) > 1e-6,
    )
    tangent_east = east * log_scale
    tangent_north = north * log_scale
    return np.stack(
        [distance, cosine, tangent_east, tangent_north], axis=0
    ).astype(np.float32)


class EarlyHorizonDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        *,
        split: str,
        target_mode: str = "correct",
        shuffle_seed: int = 20260711,
        allow_outer_test: bool = True,
        tile_rows: int = 8,
        tile_cols: int = 16,
        consensus_root: str | Path | None = None,
    ) -> None:
        self.root = Path(processed_root)
        self.rows = sorted(
            [row for row in read_manifest(self.root / "manifest.jsonl") if row["split"] == split],
            key=lambda row: str(row["image_id"]),
        )
        if not self.rows:
            raise ValueError(f"No early-horizon rows for split={split}")
        # ``allow_outer_test`` is retained as a compatibility argument.  Test
        # targets are ordinary evaluation data in the standard 59/13/13
        # protocol; checkpoint and hyperparameter selection remains val-only.
        del allow_outer_test
        mode = str(target_mode).lower()
        if mode not in {"correct", "shuffled"}:
            raise ValueError("target_mode must be correct or shuffled")
        self.split = str(split)
        self.target_mode = mode if split == "train" else "correct"
        self.shuffle_seed = int(shuffle_seed)
        self.tile_rows = int(tile_rows)
        self.tile_cols = int(tile_cols)
        self.consensus_root = Path(consensus_root) if consensus_root is not None else None
        if self.tile_rows <= 0 or self.tile_cols <= 0:
            raise ValueError("tile grid must be positive")
        self.template_root = self.root / "baselines" / "early_template"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        image_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        height, width = image_t.shape[-2:]
        with np.load(row["target"]) as target_data:
            target = np.asarray(target_data["cumulative"], dtype=np.float32)
            final = np.asarray(target_data["final"], dtype=np.float32)
            fixation = np.asarray(target_data["fixation"], dtype=np.float32)
            area = np.asarray(target_data["area_weights"], dtype=np.float32)
            horizons = np.asarray(target_data["horizons_s"], dtype=np.float32)
        with np.load(self.template_root / f"{row['image_id']}.npz") as template_data:
            template = np.asarray(template_data["cumulative"], dtype=np.float32)
            template_horizons = np.asarray(template_data["horizons_s"], dtype=np.float32)
        if not np.allclose(horizons, template_horizons):
            raise ValueError(f"Target/template horizon mismatch for {row['image_id']}")
        permutation = None
        if self.target_mode == "shuffled":
            permutation = _derangement(len(horizons), str(row["image_id"]), self.shuffle_seed)
            target = target[permutation]
        with np.load(row["fixations"]) as fixation_data:
            raw_fixations = np.asarray(fixation_data["fixations"], dtype=np.float32)
        observer_ids = np.unique(raw_fixations[:, 3]).astype(int).tolist()
        raw_dwell_tiles = np.stack(
            [
                raw_dwell_tile_distribution(
                    raw_fixations,
                    observer_ids=observer_ids,
                    horizon_s=float(horizon),
                    tile_rows=self.tile_rows,
                    tile_cols=self.tile_cols,
                )
                for horizon in horizons
            ],
            axis=0,
        )
        if permutation is not None:
            raw_dwell_tiles = raw_dwell_tiles[permutation]
        sample = {
            "image": torch.cat([image_t, coord_channels(height, width)], dim=0),
            "start_geometry": torch.from_numpy(
                _declared_start_geometry(row, height, width)
            ),
            "target": torch.from_numpy(target.copy()),
            "final": torch.from_numpy(final.copy()).unsqueeze(0),
            "fixation": torch.from_numpy(fixation.copy()).unsqueeze(0),
            "template": torch.from_numpy(template.copy()),
            "area_weights": torch.from_numpy(area).unsqueeze(0),
            "horizons_s": torch.from_numpy(horizons),
            "image_id": str(row["image_id"]),
            "canonical_content_id": str(row["canonical_content_id"]),
            "fixations_path": str(row["fixations"]),
            "raw_dwell_tiles": torch.from_numpy(raw_dwell_tiles.copy()),
        }
        if self.consensus_root is not None:
            with np.load(self.consensus_root / f"{row['image_id']}.npz") as consensus_data:
                consensus = np.asarray(consensus_data["topk_consensus"], dtype=np.float32)
                consensus_horizons = np.asarray(consensus_data["horizons_s"], dtype=np.float32)
            if not np.allclose(horizons, consensus_horizons):
                raise ValueError(f"Consensus horizon mismatch for {row['image_id']}")
            if permutation is not None:
                consensus = consensus[permutation]
            sample["raw_topk_consensus"] = torch.from_numpy(consensus.copy())
        return sample
