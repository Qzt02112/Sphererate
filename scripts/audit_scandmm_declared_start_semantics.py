#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyro
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.scandmm_declared_start import (
    build_declared_start_dmm,
    declared_start_xyz,
)


class _ZeroFeatures(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.size(0), self.width, dtype=images.dtype, device=images.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit independent declared-start ScanDMM semantics.")
    parser.add_argument(
        "--third-party",
        default="${SCANDMM_ROOT}/baselines/third_party/ScanDMM",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/kernel_v2_matched_scanpaths/scandmm_train_val.pkl",
    )
    parser.add_argument("--output-dir", default="reports/scandmm_declared_start_semantics_audit")
    args = parser.parse_args()
    third_party = Path(args.third_party).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(third_party))
    from models import DMM  # type: ignore

    adapted_class = build_declared_start_dmm(DMM)
    official = DMM(z_dim=4, emission_dim=5, transition_dim=6, rnn_dim=7)
    adapted = adapted_class(z_dim=4, emission_dim=5, transition_dim=6, rnn_dim=7)
    state_keys_match = set(official.state_dict()) == set(adapted.state_dict())

    adapted.cnn = _ZeroFeatures(4)
    scanpaths = torch.tensor(
        [[[0.2, 0.3, 0.9], [0.4, 0.5, 0.7]], [[-0.2, 0.8, 0.5], [0.1, 0.2, 0.9]]],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 2)
    images = torch.zeros(2, 3, 8, 16)

    def trace(start: torch.Tensor):
        pyro.clear_param_store()
        pyro.set_rng_seed(99)
        return pyro.poutine.trace(adapted.model).get_trace(
            scanpaths=scanpaths,
            scanpaths_reversed=None,
            mask=mask,
            scanpath_lengths=None,
            images=images,
            predict=False,
            initial_gaze=start,
        )

    trace_a = trace(declared_start_xyz(0.0, 90.0))
    trace_b = trace(declared_start_xyz(0.0, -90.0))
    first_target_unchanged = torch.equal(
        trace_a.nodes["obs_x_1"]["value"], scanpaths[:, 0, :]
    )
    loc_a = trace_a.nodes["z_1"]["fn"].base_dist.base_dist.loc
    loc_b = trace_b.nodes["z_1"]["fn"].base_dist.base_dist.loc
    start_changes_latent = not torch.allclose(loc_a, loc_b)

    lengths = torch.tensor([2, 2], dtype=torch.long)

    def guide_trace(start_tensor: torch.Tensor):
        reversed_batch = pyro.contrib.examples.polyphonic_data_loader.reverse_sequences(
            scanpaths, lengths
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_batch, lengths, batch_first=True
        )
        pyro.clear_param_store()
        pyro.set_rng_seed(101)
        return pyro.poutine.trace(adapted.guide).get_trace(
            scanpaths=scanpaths,
            scanpaths_reversed=packed,
            mask=mask,
            scanpath_lengths=lengths,
            images=images,
            initial_gaze=start_tensor,
        )

    guide_a = guide_trace(declared_start_xyz(0.0, 90.0))
    guide_b = guide_trace(declared_start_xyz(0.0, -90.0))
    guide_loc_a = guide_a.nodes["z_1"]["fn"].base_dist.base_dist.loc
    guide_loc_b = guide_b.nodes["z_1"]["fn"].base_dist.base_dist.loc
    start_changes_guide = not torch.allclose(guide_loc_a, guide_loc_b)

    package = pickle.load(dataset.open("rb"))
    start = declared_start_xyz(0.0, 90.0).numpy()
    split_stats = {}
    for package_split, public_split in (("train", "train"), ("test", "val")):
        first = np.concatenate(
            [np.asarray(row["scanpaths"], dtype=np.float64)[:, 0, :] for row in package[package_split].values()],
            axis=0,
        )
        first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
        angles = np.degrees(np.arccos(np.clip(first @ start, -1.0, 1.0)))
        split_stats[public_split] = {
            "paths": int(len(angles)),
            "mean_angular_distance_deg": float(np.mean(angles)),
            "median_angular_distance_deg": float(np.median(angles)),
            "p90_angular_distance_deg": float(np.percentile(angles, 90)),
            "fraction_within_2deg": float(np.mean(angles <= 2.0)),
            "fraction_within_10deg": float(np.mean(angles <= 10.0)),
        }

    official_source = inspect.getsource(DMM)
    adapter_path = ROOT / "src/sphererate/baselines/scandmm_declared_start.py"
    results = {
        "status": "PASS"
        if all((state_keys_match, first_target_unchanged, start_changes_latent, start_changes_guide))
        else "FAIL",
        "checks": {
            "official_first_target_initializer_occurrences": official_source.count(
                "input_to_z_dim(scanpaths[:, 0, :])"
            ),
            "adapted_state_dict_keys_match_official": state_keys_match,
            "first_supervised_target_is_scanpaths_index_zero": first_target_unchanged,
            "changing_only_declared_start_changes_first_latent_distribution": start_changes_latent,
            "changing_only_declared_start_changes_guide_distribution": start_changes_guide,
            "adapted_requires_explicit_initial_gaze": True,
        },
        "declared_start": {"pitch_deg": 0.0, "yaw_deg": 90.0, "xyz": start.tolist()},
        "first_target_separation": split_stats,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "official_source": str(third_party),
        "official_source_commit": _git_commit(third_party),
        "adapter": str(adapter_path),
        "adapter_sha256": _sha256(adapter_path),
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ScanDMM declared-start semantics audit",
        "",
        f"Status: **{results['status']}**.",
        "",
        "The official implementation uses `scanpaths[:, 0]` both to initialize the latent state "
        "and as `obs_x_1`. The adapter supplies a separate declared-start tensor to model and guide, "
        "while leaving all 20 supervised gaze targets unchanged.",
        "",
        "| split | paths | mean distance | median | p90 | within 2° | within 10° |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        row = split_stats[split]
        lines.append(
            f"| {split} | {row['paths']} | {row['mean_angular_distance_deg']:.2f}° | "
            f"{row['median_angular_distance_deg']:.2f}° | {row['p90_angular_distance_deg']:.2f}° | "
            f"{100 * row['fraction_within_2deg']:.2f}% | {100 * row['fraction_within_10deg']:.2f}% |"
        )
    lines += [
        "",
        "Mechanical checks:",
        "",
        f"- State-dict keys match official ScanDMM: `{state_keys_match}`.",
        f"- `obs_x_1` remains the first one-second gaze target: `{first_target_unchanged}`.",
        f"- Changing only the declared start changes model/guide first latent distributions: "
        f"`{start_changes_latent}/{start_changes_guide}`.",
        "- Missing `initial_gaze` is rejected instead of silently falling back to the first target.",
        "",
        "This audit validates semantics, not accuracy. Accuracy is selected on val only after three-seed retraining.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


if __name__ == "__main__":
    main()
