#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pyro
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.matched_protocol import load_matched_protocol
from sphererate.baselines.scandmm_declared_start import (
    build_declared_start_dmm,
    build_declared_start_trainer,
)


class SparseCheckpointTrain:
    """Factory for an official ScanDMM trainer with bounded checkpoint storage."""

    @staticmethod
    def build(official_train_class: type, checkpoint_every: int, output: Path) -> type:
        class Trainer(official_train_class):
            def save_checkpoint(self, name: str) -> None:
                match = re.search(r"epoch-(\d+)\.pkl$", name)
                if match is None:
                    raise ValueError(f"Cannot parse ScanDMM checkpoint name: {name}")
                epoch = int(match.group(1))
                final_epoch = int(self.args.epochs) - 1
                if (epoch + 1) % int(checkpoint_every) != 0 and epoch != final_epoch:
                    return
                checkpoint = output / "checkpoints" / f"epoch_{epoch:04d}.pkl"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.dmm.state_dict(), checkpoint)

        return Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain official ScanDMM on the frozen kernel-v2 train split.")
    parser.add_argument("--protocol", default="configs/kernel_v2_matched_baselines.yaml")
    parser.add_argument("--dataset", default="data/processed/kernel_v2_matched_scanpaths/scandmm_train_val.pkl")
    parser.add_argument("--third-party", default="${SCANDMM_ROOT}/baselines/third_party/ScanDMM")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-dir")
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-val", type=int)
    args = parser.parse_args()
    protocol = load_matched_protocol(args.protocol)
    if int(args.seed) not in protocol.seeds:
        raise ValueError(f"Seed {args.seed} is not frozen in the matched protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("Matched ScanDMM retraining requires CUDA")
    dataset = Path(args.dataset).resolve()
    third_party = Path(args.third_party).resolve()
    output = Path(args.run_dir or f"runs/kernel_v2_scandmm_declared_start_seed{args.seed}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(third_party))
    import config as official_config  # type: ignore
    from models import DMM  # type: ignore
    from train import Train as OfficialTrain  # type: ignore
    import pickle

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    pyro.set_rng_seed(int(args.seed))
    pyro.clear_param_store()
    package = pickle.load(dataset.open("rb"))
    expected_train = int(
        args.expected_train
        if args.expected_train is not None
        else protocol.expected_content_counts[protocol.train_split]
    )
    expected_val = int(
        args.expected_val
        if args.expected_val is not None
        else protocol.expected_content_counts[protocol.selection_split]
    )
    if package["info"]["train"]["num_image"] != expected_train or package["info"]["test"]["num_image"] != expected_val:
        raise RuntimeError(
            "ScanDMM pickle does not match the declared split: "
            f"found {package['info']['train']['num_image']}/{package['info']['test']['num_image']}, "
            f"expected {expected_train}/{expected_val}"
        )
    trainer_args = argparse.Namespace(
        dataset=str(dataset),
        lr=float(official_config.learning_rate),
        bs=int(args.batch_size),
        epochs=int(args.epochs),
        clip_norm=10.0,
        lr_decay=float(official_config.lr_decay),
        weight_decay=float(official_config.weight_decay),
        annealing_epochs=int(official_config.annealing_epochs),
        minimum_annealing_factor=float(official_config.minimum_annealing_factor),
        load_model=None,
        load_opt=None,
        save_root=str(output / "checkpoints") + "/",
    )
    declared_start_dmm = build_declared_start_dmm(DMM)
    declared_start_train = build_declared_start_trainer(
        OfficialTrain,
        pitch_deg=float(args.start_pitch_deg),
        yaw_deg=float(args.start_yaw_deg),
    )
    model = declared_start_dmm(use_cuda=True)
    trainer_class = SparseCheckpointTrain.build(
        declared_start_train, int(args.checkpoint_every), output
    )
    trainer = trainer_class(model, package, trainer_args, str(output / "train.log"))
    metadata = {
        "method": "ScanDMM-declared-start-v3",
        "conditioning_semantics": {
            "initial_gaze": "independent_declared_tensor",
            "declared_start_pitch_deg": float(args.start_pitch_deg),
            "declared_start_yaw_deg": float(args.start_yaw_deg),
            "first_supervised_target": "scanpaths[:, 0, :] (unchanged)",
            "supervised_sequence_length": int(package["info"]["train"]["scanpath_length"]),
            "legacy_first_target_as_initializer": False,
        },
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "checkpoint_every": int(args.checkpoint_every),
        "batch_size": int(args.batch_size),
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source": str(third_party),
        "source_commit": _git_commit(third_party),
        "train_contents": int(package["info"]["train"]["num_image"]),
        "val_contents": int(package["info"]["test"]["num_image"]),
        "selection": "pending_val_early_macro_kld_over_sparse_checkpoints",
        "gate_used_for_training_or_selection": False,
        "test_target_rows_loaded": 0,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    trainer.run()
    checkpoints = sorted((output / "checkpoints").glob("epoch_*.pkl"))
    if not checkpoints:
        raise RuntimeError("ScanDMM training produced no sparse checkpoints")
    metadata["checkpoints"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in checkpoints
    ]
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(output), "checkpoints": len(checkpoints)}, indent=2))


def _git_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value[5:]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return value


if __name__ == "__main__":
    main()
