from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sphererate.data.conditioning import apply_prefix_condition_np, target_npz_to_dict
from sphererate.data.maps import MapGeometry, erp_gaussian_map, fixation_binary_map
from sphererate.utils import read_jsonl


def coord_channels(height: int, width: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width).view(1, width).repeat(height, 1)
    y = torch.linspace(0.0, 1.0, height).view(height, 1).repeat(1, width)
    lon = 2 * torch.pi * x - torch.pi
    lat = torch.pi / 2 - torch.pi * y
    equator = torch.cos(lat).clamp_min(0.0)
    return torch.stack(
        [
            torch.sin(lon),
            torch.cos(lon),
            torch.sin(lat),
            torch.cos(lat),
            equator,
        ],
        dim=0,
    ).float()


class ProcessedSaliencyDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        augment_roll: bool = False,
        augment_roll_degrees: Sequence[float] | None = None,
        max_samples: int | None = None,
        image_id_file: str | Path | None = None,
        shuffle_temporal_labels: bool = False,
        shuffle_seed: int = 20260708,
        temporal_target_mode: str = "correct",
        condition_prefix_bins: int = 0,
        condition_prefix_step_channel: bool = False,
        condition_startpoint_channels: bool = False,
        startpoint_condition_mode: str = "actual",
        startpoint_prior_sigma_px: float = 10.0,
        startpoint_random_seed: int = 20260710,
    ) -> None:
        self.processed_root = Path(processed_root)
        rows = read_jsonl(self.processed_root / "manifest.jsonl")
        self.rows = [row for row in rows if row["split"] == split]
        if image_id_file:
            allowed = _read_image_ids(image_id_file)
            self.rows = [row for row in self.rows if str(row["image_id"]) in allowed]
        self.rows.sort(key=lambda row: row["image_id"])
        if max_samples is not None:
            self.rows = _limit_rows_balanced(self.rows, int(max_samples))
        if not self.rows:
            suffix = f" with ids from {image_id_file}" if image_id_file else ""
            raise ValueError(f"No rows found for split={split}{suffix}")
        self.split = split
        self.augment_roll = augment_roll and split == "train"
        self.augment_roll_degrees = [float(x) for x in (augment_roll_degrees or [])]
        target_mode = str(temporal_target_mode).lower()
        if target_mode not in {"correct", "shuffled", "final_repeat"}:
            raise ValueError(f"Unsupported temporal_target_mode: {temporal_target_mode}")
        if shuffle_temporal_labels:
            if target_mode not in {"correct", "shuffled"}:
                raise ValueError("shuffle_temporal_labels conflicts with temporal_target_mode")
            target_mode = "shuffled"
        self.temporal_target_mode = target_mode if split == "train" else "correct"
        self.shuffle_temporal_labels = self.temporal_target_mode == "shuffled"
        self.shuffle_seed = int(shuffle_seed)
        self.condition_prefix_bins = int(condition_prefix_bins)
        self.condition_prefix_step_channel = bool(condition_prefix_step_channel)
        self.condition_startpoint_channels = bool(condition_startpoint_channels)
        self.startpoint_condition_mode = str(startpoint_condition_mode).lower()
        if self.startpoint_condition_mode not in {"actual", "swapped", "counterpart", "random", "zero"}:
            raise ValueError(f"Unsupported startpoint_condition_mode: {startpoint_condition_mode}")
        self.startpoint_prior_sigma_px = float(startpoint_prior_sigma_px)
        self.startpoint_random_seed = int(startpoint_random_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        with Image.open(row["image"]) as image:
            img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        h, w = img_t.shape[-2:]

        with np.load(row["target"]) as target_data:
            target = _saliency_target_npz_to_dict(target_data, row)

        prefix_channels: torch.Tensor | None = None
        if self.condition_prefix_bins > 0:
            fixations = _load_fixations(row.get("fixations"))
            target, prefix_np = apply_prefix_condition_np(
                target,
                self.condition_prefix_bins,
                fixations=fixations,
                include_step_channel=self.condition_prefix_step_channel,
            )
            prefix_channels = torch.from_numpy(prefix_np.astype(np.float32))

        temporal = torch.from_numpy(target["temporal"].astype(np.float32))
        cumulative = torch.from_numpy(target["cumulative"].astype(np.float32))
        final = torch.from_numpy(target["final"].astype(np.float32)).unsqueeze(0)
        fixation = torch.from_numpy(target["fixation"].astype(np.float32)).unsqueeze(0)
        counterpart_final = _counterpart_final_tensor(row, target)
        area = torch.from_numpy(target["area_weights"].astype(np.float32)).unsqueeze(0)
        startpoint_channels: torch.Tensor | None = None
        if self.condition_startpoint_channels:
            startpoint_channels = torch.from_numpy(
                _startpoint_relative_channels(
                    _startpoint_row(row, self.startpoint_condition_mode, self.startpoint_random_seed),
                    h,
                    w,
                    target["area_weights"].astype(np.float32),
                    sigma_px=self.startpoint_prior_sigma_px,
                )
            )

        if self.shuffle_temporal_labels:
            stable_id = sum(ord(ch) for ch in str(row["image_id"]))
            generator = torch.Generator().manual_seed(self.shuffle_seed + stable_id)
            perm = torch.randperm(temporal.shape[0], generator=generator)
            temporal = temporal[perm]
            cumulative = _cumulative_from_temporal_gt(temporal, area)
        elif self.temporal_target_mode == "final_repeat":
            temporal = final.repeat(temporal.shape[0], 1, 1)
            cumulative = temporal.clone()

        if self.augment_roll:
            shift = _sample_yaw_shift(w, self.augment_roll_degrees)
            if shift:
                img_t = torch.roll(img_t, shifts=shift, dims=-1)
                temporal = torch.roll(temporal, shifts=shift, dims=-1)
                cumulative = torch.roll(cumulative, shifts=shift, dims=-1)
                final = torch.roll(final, shifts=shift, dims=-1)
                if counterpart_final is not None:
                    counterpart_final = torch.roll(counterpart_final, shifts=shift, dims=-1)
                fixation = torch.roll(fixation, shifts=shift, dims=-1)
                if prefix_channels is not None:
                    prefix_channels = torch.roll(prefix_channels, shifts=shift, dims=-1)
                if startpoint_channels is not None:
                    startpoint_channels = torch.roll(startpoint_channels, shifts=shift, dims=-1)

        coords = coord_channels(h, w)
        parts = [img_t, coords]
        if prefix_channels is not None:
            parts.append(prefix_channels)
        if startpoint_channels is not None:
            parts.append(startpoint_channels)
        x = torch.cat(parts, dim=0)

        item = {
            "image": x,
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "fixation": fixation,
            "area_weights": area,
            "image_id": row["image_id"],
            "dataset": _row_dataset(row),
        }
        if counterpart_final is not None:
            item["counterpart_final"] = counterpart_final
        return item


class ProcessedNextFixationDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        prefix_length: int = 3,
        prefix_mode: str = "normal",
        prefix_encoding: str = "collapsed",
        negative_prefix_mode: str | Sequence[str] | None = None,
        augment_roll: bool = False,
        augment_roll_degrees: Sequence[float] | None = None,
        max_samples: int | None = None,
        image_id_file: str | Path | None = None,
        sample_id_file: str | Path | None = None,
        sigma_px: float = 10.0,
        area_weighted: bool = True,
        shuffle_seed: int = 20260708,
        start_geometry: bool = False,
    ) -> None:
        self.processed_root = Path(processed_root)
        image_rows = [row for row in read_jsonl(self.processed_root / "manifest.jsonl") if row["split"] == split]
        if image_id_file:
            allowed = _read_image_ids(image_id_file)
            image_rows = [row for row in image_rows if str(row["image_id"]) in allowed]
        image_rows.sort(key=lambda row: row["image_id"])
        self.prefix_length = int(prefix_length)
        if self.prefix_length < 1:
            raise ValueError(f"scanpath_prefix_length must be >= 1, got {self.prefix_length}")
        self.prefix_mode = str(prefix_mode).lower()
        self._allowed_prefix_modes = {
            "normal",
            "reversed",
            "shuffled",
            "none",
            "history_reversed",
            "history_shuffled",
            "current_only",
            "start_only",
            "start_repeat",
        }
        if self.prefix_mode not in self._allowed_prefix_modes:
            raise ValueError(f"Unsupported scanpath_prefix_mode: {prefix_mode}")
        if negative_prefix_mode is None or negative_prefix_mode == "":
            self.negative_prefix_modes: list[str] = []
        elif isinstance(negative_prefix_mode, str):
            self.negative_prefix_modes = [negative_prefix_mode.lower()]
        else:
            self.negative_prefix_modes = [str(mode).lower() for mode in negative_prefix_mode if str(mode)]
        for negative_mode in self.negative_prefix_modes:
            if negative_mode not in self._allowed_prefix_modes:
                raise ValueError(f"Unsupported scanpath_negative_prefix_mode: {negative_mode}")
        self.negative_prefix_mode = self.negative_prefix_modes[0] if len(self.negative_prefix_modes) == 1 else None
        self.prefix_encoding = str(prefix_encoding).lower()
        if self.prefix_encoding not in {"collapsed", "sequence"}:
            raise ValueError(f"Unsupported scanpath_prefix_encoding: {prefix_encoding}")
        self.split = split
        self.augment_roll = augment_roll and split == "train"
        self.augment_roll_degrees = [float(x) for x in (augment_roll_degrees or [])]
        self.sigma_px = float(sigma_px)
        self.area_weighted = bool(area_weighted)
        self.shuffle_seed = int(shuffle_seed)
        self.start_geometry = bool(start_geometry)
        self.rows = self._build_sample_rows(image_rows)
        if sample_id_file:
            allowed_samples = _read_image_ids(sample_id_file)
            self.rows = [row for row in self.rows if str(row["sample_id"]) in allowed_samples]
        if max_samples is not None:
            self.rows = _limit_rows_balanced(self.rows, int(max_samples), seed=self.shuffle_seed)
        if not self.rows:
            filters = []
            if image_id_file:
                filters.append(f"image ids from {image_id_file}")
            if sample_id_file:
                filters.append(f"sample ids from {sample_id_file}")
            suffix = f" with {' and '.join(filters)}" if filters else ""
            raise ValueError(f"No next-fixation samples found for split={split}{suffix}")

    def _build_sample_rows(self, image_rows: list[dict]) -> list[dict]:
        sample_rows: list[dict] = []
        for row in image_rows:
            fix = _load_fixations(row.get("fixations"))
            if fix is None or fix.ndim != 2 or fix.shape[1] < 6:
                continue
            by_observer: dict[int, list[np.ndarray]] = defaultdict(list)
            for fixation in fix:
                by_observer[int(fixation[3])].append(fixation)
            for observer_id, points in by_observer.items():
                ordered = np.asarray(sorted(points, key=lambda item: (float(item[2]), float(item[5]))), dtype=np.float32)
                if len(ordered) <= self.prefix_length:
                    continue
                for target_index in range(self.prefix_length, len(ordered)):
                    sample_rows.append(
                        {
                            **row,
                            "observer_id": int(observer_id),
                            "target_index": int(target_index),
                            "sample_id": f"{row['image_id']}__obs{observer_id}__t{target_index}",
                        }
                    )
        sample_rows.sort(key=lambda row: (row["image_id"], int(row["observer_id"]), int(row["target_index"])))
        return sample_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        with Image.open(row["image"]) as image:
            img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        h, w = img_t.shape[-2:]
        geom = MapGeometry(h, w)
        area_np = geom.area_weights

        fix = _load_fixations(row.get("fixations"))
        if fix is None:
            raise ValueError(f"Missing fixations for {row['image_id']}")
        obs = fix[fix[:, 3].astype(np.int64) == int(row["observer_id"])]
        obs = np.asarray(sorted(obs, key=lambda item: (float(item[2]), float(item[5]))), dtype=np.float32)
        target_index = int(row["target_index"])
        prefix = obs[target_index - self.prefix_length : target_index]
        target = obs[target_index]

        start_point = obs[:1]
        prefix_channels = self._prefix_channels(prefix, row, self.prefix_mode, h, w, area_np, start_point=start_point)
        next_map = self._points_map(target[None, :], h, w, area_np)
        fixation_map = fixation_binary_map(target[None, 0], target[None, 1], h, w)
        step_value = float(target_index) / float(max(len(obs) - 1, 1))
        step_map = np.full((h, w), step_value, dtype=np.float32)

        prefix_t = torch.from_numpy(prefix_channels)
        start_geometry_np = self._start_geometry_channels(self.prefix_mode, start_point, h, w)
        start_geometry_t = torch.from_numpy(start_geometry_np)
        negative_prefix_ts: list[torch.Tensor] = []
        negative_start_geometry_ts: list[torch.Tensor] = []
        for negative_mode in self.negative_prefix_modes:
            negative_prefix_channels = self._prefix_channels(
                prefix,
                row,
                negative_mode,
                h,
                w,
                area_np,
                start_point=start_point,
            )
            negative_prefix_ts.append(torch.from_numpy(negative_prefix_channels))
            negative_start_geometry_ts.append(torch.from_numpy(self._start_geometry_channels(negative_mode, start_point, h, w)))
        step_t = torch.from_numpy(step_map.astype(np.float32)).unsqueeze(0)
        final = torch.from_numpy(next_map.astype(np.float32)).unsqueeze(0)
        fixation = torch.from_numpy(fixation_map.astype(np.float32)).unsqueeze(0)
        area = torch.from_numpy(area_np.astype(np.float32)).unsqueeze(0)
        temporal = final.clone()
        cumulative = final.clone()

        if self.augment_roll:
            shift = _sample_yaw_shift(w, self.augment_roll_degrees)
            if shift:
                img_t = torch.roll(img_t, shifts=shift, dims=-1)
                prefix_t = torch.roll(prefix_t, shifts=shift, dims=-1)
                start_geometry_t = torch.roll(start_geometry_t, shifts=shift, dims=-1)
                negative_prefix_ts = [torch.roll(negative_prefix_t, shifts=shift, dims=-1) for negative_prefix_t in negative_prefix_ts]
                negative_start_geometry_ts = [
                    torch.roll(negative_start_geometry_t, shifts=shift, dims=-1)
                    for negative_start_geometry_t in negative_start_geometry_ts
                ]
                final = torch.roll(final, shifts=shift, dims=-1)
                fixation = torch.roll(fixation, shifts=shift, dims=-1)
                temporal = torch.roll(temporal, shifts=shift, dims=-1)
                cumulative = torch.roll(cumulative, shifts=shift, dims=-1)

        parts = [img_t, coord_channels(h, w), prefix_t, step_t]
        if self.start_geometry:
            parts.append(start_geometry_t)
        x = torch.cat(parts, dim=0)
        sample = {
            "image": x,
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "fixation": fixation,
            "area_weights": area,
            "image_id": str(row["sample_id"]),
            "source_image_id": str(row["image_id"]),
            "dataset": _row_dataset(row),
        }
        if negative_prefix_ts:
            negative_images = [
                torch.cat(
                    [
                        img_t,
                        coord_channels(h, w),
                        negative_prefix_t,
                        step_t,
                        negative_start_geometry_t,
                    ]
                    if self.start_geometry
                    else [img_t, coord_channels(h, w), negative_prefix_t, step_t],
                    dim=0,
                )
                for negative_prefix_t, negative_start_geometry_t in zip(negative_prefix_ts, negative_start_geometry_ts)
            ]
            if len(negative_images) == 1:
                sample["negative_image"] = negative_images[0]
            sample["negative_images"] = torch.stack(negative_images, dim=0)
        return sample

    def _start_geometry_channels(self, mode: str, start_point: np.ndarray | None, height: int, width: int) -> np.ndarray:
        channels = np.zeros((3, height, width), dtype=np.float32)
        if not self.start_geometry or mode not in {"start_only", "start_repeat"}:
            return channels
        if start_point is None or len(start_point) == 0:
            return channels

        x = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
        y = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
        lon = (2.0 * np.pi * x - np.pi)[None, :]
        lat = (np.pi / 2.0 - np.pi * y)[:, None]
        start_lon = 2.0 * np.pi * float(start_point[0, 0] % 1.0) - np.pi
        start_lat = np.pi / 2.0 - np.pi * float(np.clip(start_point[0, 1], 0.0, 1.0))
        dlon = lon - start_lon
        channels[0] = np.sin(dlon)
        channels[1] = np.cos(dlon)
        channels[2] = (lat - start_lat) / np.pi
        return channels.astype(np.float32)

    def _prefix_channels(
        self,
        prefix: np.ndarray,
        row: dict,
        mode: str,
        height: int,
        width: int,
        area_weights: np.ndarray,
        start_point: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.prefix_encoding == "sequence":
            prefix_maps = [
                self._points_map(point, height, width, area_weights)
                for point in self._sequence_prefix_points(prefix, row, mode, start_point=start_point)
            ]
            return np.stack(prefix_maps, axis=0).astype(np.float32)
        if mode == "current_only":
            transformed_prefix = prefix[:0]
            current = prefix[-1:] if len(prefix) else prefix
        elif mode == "start_only":
            transformed_prefix = start_point if start_point is not None else prefix[:1]
            current = transformed_prefix
        elif mode == "start_repeat":
            transformed_prefix = np.repeat(start_point if start_point is not None else prefix[:1], self.prefix_length, axis=0)
            current = (start_point if start_point is not None else prefix[:1])
        else:
            transformed_prefix = self._transform_prefix(prefix, row, mode, start_point=start_point)
            current = transformed_prefix[-1:] if len(transformed_prefix) else transformed_prefix
        prefix_map = self._points_map(transformed_prefix, height, width, area_weights)
        current_map = self._points_map(current, height, width, area_weights)
        return np.stack([prefix_map, current_map], axis=0).astype(np.float32)

    def _points_map(self, points: np.ndarray, height: int, width: int, area_weights: np.ndarray) -> np.ndarray:
        if self.prefix_mode == "none" and len(points) > 1:
            points = points[:0]
        if len(points) == 0:
            return np.zeros((height, width), dtype=np.float32)
        return erp_gaussian_map(
            points[:, 0],
            points[:, 1],
            height,
            width,
            sigma_px=self.sigma_px,
            weights=points[:, 4],
            area_weights=area_weights,
            area_weighted=self.area_weighted,
        )

    def _transform_prefix(
        self,
        prefix: np.ndarray,
        row: dict,
        mode: str | None = None,
        start_point: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = self.prefix_mode if mode is None else mode
        if mode == "normal":
            return prefix
        if mode == "none":
            return prefix[:0]
        if mode == "current_only":
            return prefix[-1:].copy() if len(prefix) else prefix
        if mode == "start_only":
            return (start_point if start_point is not None else prefix[:1]).copy()
        if mode == "start_repeat":
            start = start_point if start_point is not None else prefix[:1]
            return np.repeat(start, self.prefix_length, axis=0).copy()
        if mode == "reversed":
            return prefix[::-1].copy()
        if mode == "history_reversed":
            if len(prefix) <= 1:
                return prefix.copy()
            return np.concatenate([prefix[:-1][::-1], prefix[-1:]], axis=0).copy()
        if mode == "history_shuffled":
            if len(prefix) <= 2:
                return prefix.copy()
            stable_id = sum(ord(ch) for ch in str(row["sample_id"]))
            rng = np.random.default_rng(self.shuffle_seed + stable_id)
            history = prefix[:-1][rng.permutation(len(prefix) - 1)]
            return np.concatenate([history, prefix[-1:]], axis=0).copy()
        stable_id = sum(ord(ch) for ch in str(row["sample_id"]))
        rng = np.random.default_rng(self.shuffle_seed + stable_id)
        return prefix[rng.permutation(len(prefix))].copy()

    def _sequence_prefix_points(
        self,
        prefix: np.ndarray,
        row: dict,
        mode: str | None = None,
        start_point: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        mode = self.prefix_mode if mode is None else mode
        empty = prefix[:0]
        if mode == "current_only":
            points = [empty for _ in range(self.prefix_length)]
            if len(prefix):
                points[-1] = prefix[-1:].copy()
            return points
        if mode == "start_only":
            points = [empty for _ in range(self.prefix_length)]
            start = start_point if start_point is not None else prefix[:1]
            if len(start):
                points[-1] = start.copy()
            return points
        if mode == "start_repeat":
            start = start_point if start_point is not None else prefix[:1]
            return [start.copy() for _ in range(self.prefix_length)]
        transformed = self._transform_prefix(prefix, row, mode, start_point=start_point)
        points = []
        for prefix_idx in range(self.prefix_length):
            point = transformed[prefix_idx : prefix_idx + 1] if prefix_idx < len(transformed) else empty
            points.append(point)
        return points


def _row_dataset(row: dict) -> str:
    return str(row.get("dataset") or "unknown")


def _read_image_ids(path: str | Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _load_fixations(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    with np.load(path) as data:
        if "fixations" not in data:
            return None
        return data["fixations"].astype(np.float32)


def _startpoint_relative_channels(
    row: dict,
    height: int,
    width: int,
    area_weights: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    if "start_lon_deg" not in row:
        raise KeyError(f"Row {row.get('image_id')} has no start_lon_deg for startpoint conditioning.")
    start_lon = np.deg2rad(float(row["start_lon_deg"]))
    start_lat = np.deg2rad(float(row.get("start_lat_deg", 0.0)))
    x = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    y = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lon = (2.0 * np.pi * x - np.pi)[None, :]
    lat = (np.pi / 2.0 - np.pi * y)[:, None]
    dlon = lon - start_lon
    proximity = np.sin(lat) * np.sin(start_lat) + np.cos(lat) * np.cos(start_lat) * np.cos(dlon)
    start_x = ((float(row["start_lon_deg"]) + 180.0) / 360.0) % 1.0
    start_y = np.clip((90.0 - float(row.get("start_lat_deg", 0.0))) / 180.0, 0.0, np.nextafter(1.0, 0.0))
    start_prior = erp_gaussian_map(
        np.asarray([start_x], dtype=np.float32),
        np.asarray([start_y], dtype=np.float32),
        height,
        width,
        sigma_px=sigma_px,
        area_weights=area_weights,
        area_weighted=True,
    )
    return np.stack(
        [
            np.sin(dlon).repeat(height, axis=0),
            np.cos(dlon).repeat(height, axis=0),
            np.repeat(((lat - start_lat) / np.pi), width, axis=1),
            proximity,
            start_prior,
        ],
        axis=0,
    ).astype(np.float32)


def _startpoint_row(row: dict, mode: str, random_seed: int = 20260710) -> dict:
    if mode == "actual":
        return row
    if mode in {"swapped", "counterpart"}:
        if "counterpart_start_lon_deg" not in row:
            raise KeyError(f"Row {row.get('image_id')} has no counterpart_start_lon_deg for swapped startpoint control.")
        swapped = dict(row)
        swapped["start_lon_deg"] = row["counterpart_start_lon_deg"]
        swapped["start_lat_deg"] = row.get("counterpart_start_lat_deg", 0.0)
        return swapped
    if mode == "random":
        randomized = dict(row)
        key = f"{int(random_seed)}:{row.get('image_id')}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big", signed=False)
        randomized["start_lon_deg"] = (value / float(2**64)) * 360.0 - 180.0
        randomized["start_lat_deg"] = 0.0
        return randomized
    if mode == "zero":
        zeroed = dict(row)
        zeroed["start_lon_deg"] = 0.0
        zeroed["start_lat_deg"] = 0.0
        return zeroed
    raise ValueError(f"Unsupported startpoint_condition_mode: {mode}")


def _counterpart_final_tensor(row: dict, target: dict[str, np.ndarray]) -> torch.Tensor | None:
    condition = str(row.get("condition") or "").lower()
    if condition == "good" and "bad_start_final" in target:
        return torch.from_numpy(target["bad_start_final"].astype(np.float32)).unsqueeze(0)
    if condition == "bad" and "good_start_final" in target:
        return torch.from_numpy(target["good_start_final"].astype(np.float32)).unsqueeze(0)
    return None


def _saliency_target_npz_to_dict(data: object, row: dict) -> dict[str, np.ndarray]:
    """Load only tensors used by ProcessedSaliencyDataset from potentially large NPZ files."""
    required = ("temporal", "cumulative", "final", "fixation", "area_weights")
    target = {key: data[key].astype(np.float32).squeeze() for key in required}
    condition = str(row.get("condition") or "").lower()
    counterpart_key = "bad_start_final" if condition == "good" else "good_start_final" if condition == "bad" else None
    if counterpart_key is not None and counterpart_key in data.files:
        target[counterpart_key] = data[counterpart_key].astype(np.float32).squeeze()
    return target


def _limit_rows_balanced(rows: list[dict], max_samples: int, seed: int | None = None) -> list[dict]:
    if max_samples >= len(rows):
        return rows
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[_row_dataset(row)].append(row)
    if len(by_dataset) <= 1:
        group = list(rows)
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(group)
        return group[:max_samples]

    if seed is not None:
        rng = np.random.default_rng(seed)
        by_dataset = {name: list(group) for name, group in by_dataset.items()}
        for group in by_dataset.values():
            rng.shuffle(group)

    selected: list[dict] = []
    offsets = {name: 0 for name in by_dataset}
    names = sorted(by_dataset)
    while len(selected) < max_samples:
        added = False
        for name in names:
            if len(selected) >= max_samples:
                break
            offset = offsets[name]
            group = by_dataset[name]
            if offset < len(group):
                selected.append(group[offset])
                offsets[name] = offset + 1
                added = True
        if not added:
            break
    return selected


def _sample_yaw_shift(width: int, degrees: Sequence[float]) -> int:
    if degrees:
        idx = int(torch.randint(0, len(degrees), (1,)).item())
        return int(round(width * float(degrees[idx]) / 360.0)) % width
    return int(torch.randint(0, width, (1,)).item())


def _cumulative_from_temporal_gt(temporal: torch.Tensor, area: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    cumulative = []
    running = torch.zeros_like(temporal[:1])
    area_2d = area.squeeze(0)
    for idx in range(temporal.shape[0]):
        running = running + temporal[idx : idx + 1]
        denom = (running.squeeze(0) * area_2d).sum().clamp_min(eps)
        cumulative.append((running / denom).squeeze(0))
    return torch.stack(cumulative, dim=0)
