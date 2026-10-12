#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config
from sphererate.utils import ensure_dir, read_jsonl


def _install_dummy_matplotlib() -> None:
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.image = types.SimpleNamespace(imread=lambda *args, **kwargs: None)
    cm = types.ModuleType("matplotlib.cm")
    cm.rainbow = lambda x: x
    pyplot = types.ModuleType("matplotlib.pyplot")
    for name in ["subplots", "savefig", "axis", "subplots_adjust", "clf", "close"]:
        setattr(pyplot, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.cm", cm)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)


def _prepare_inputs(processed_root: Path, split: str, input_dir: Path) -> None:
    ensure_dir(input_dir)
    rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    for row in rows:
        src = Path(row["image"]).resolve()
        dst = input_dir / f"{row['image_id']}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "gate", "test"])
    parser.add_argument("--third-party", default="${SCANDMM_ROOT}/baselines/third_party/ScanDMM")
    parser.add_argument("--model", default="${SCANDMM_ROOT}/baselines/third_party/ScanDMM/model/model_lr-0.0003_bs-64_epoch-435.pkl")
    parser.add_argument("--run-dir", default="runs/scandmm_val_quick")
    parser.add_argument(
        "--data-root",
        help="Processed early-horizon root. Overrides paths.processed_root/data.processed_name from --config.",
    )
    parser.add_argument("--n-scanpaths", type=int, default=50)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--start-yaw-deg",
        type=float,
        default=None,
        help="If set, condition every generated scanpath on this declared ERP yaw instead of a random start.",
    )
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument(
        "--independent-declared-start",
        action="store_true",
        help="Use the leak-free adapter whose initial_gaze is separate from the generated sequence.",
    )
    args = parser.parse_args()

    import torch
    import pyro

    if not torch.cuda.is_available():
        raise RuntimeError("ScanDMM inference requested GPU, but CUDA is unavailable.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pyro.set_rng_seed(args.seed)

    if args.data_root:
        processed_root = Path(args.data_root)
    else:
        config = load_config(args.config)
        processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    run_dir = ensure_dir(args.run_dir)
    input_dir = run_dir / "input"
    output_dir = run_dir / "scanpaths"
    _prepare_inputs(processed_root, args.split, input_dir)
    ensure_dir(output_dir)

    _install_dummy_matplotlib()
    third_party = Path(args.third_party).resolve()
    sys.path.insert(0, str(third_party))

    from inference import Inference  # type: ignore
    from models import DMM  # type: ignore
    from suppor_lib import sphere2xyz  # type: ignore

    if args.independent_declared_start and args.start_yaw_deg is None:
        raise ValueError("--independent-declared-start requires --start-yaw-deg")
    if args.independent_declared_start:
        from pyro.infer import Predictive
        from sphererate.baselines.scandmm_declared_start import build_declared_start_dmm

        DMM = build_declared_start_dmm(DMM)

    dmm = DMM(use_cuda=True)
    state = torch.load(args.model, map_location="cuda", weights_only=False)
    dmm.load_state_dict(state)
    dmm.eval()
    runner = Inference(
        model=dmm,
        img_path=str(input_dir),
        n_scanpaths=int(args.n_scanpaths),
        length=int(args.length),
        output_path=str(output_dir),
        if_plot=False,
    )
    if args.start_yaw_deg is not None:
        def _declared_starting_points(num_points: int) -> torch.Tensor:
            lat_lon = torch.tensor(
                [[float(args.start_pitch_deg), float(args.start_yaw_deg)]], dtype=torch.float32
            ).repeat(int(num_points), 1)
            return sphere2xyz(lat_lon)

        runner.create_random_starting_points = _declared_starting_points
    if args.independent_declared_start:
        _predict_with_independent_start(runner, Predictive)
    else:
        runner.predict()
    metadata = {
        "baseline": "ScanDMM",
        "seed": int(args.seed),
        "n_scanpaths": int(args.n_scanpaths),
        "length": int(args.length),
        "start_mode": "declared" if args.start_yaw_deg is not None else "random_official_default",
        "start_yaw_deg": None if args.start_yaw_deg is None else float(args.start_yaw_deg),
        "start_pitch_deg": None if args.start_yaw_deg is None else float(args.start_pitch_deg),
        "initialization_semantics": (
            "independent_declared_tensor"
            if args.independent_declared_start
            else "legacy_scanpaths_first_element"
        ),
        "model": str(Path(args.model).resolve()),
    }
    if args.independent_declared_start:
        adapter_path = (
            Path(__file__).resolve().parents[1]
            / "src/sphererate/baselines/scandmm_declared_start.py"
        )
        metadata["declared_start_adapter"] = {
            "path": str(adapter_path),
            "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        }
    (run_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved ScanDMM scanpaths to {output_dir} (seed={args.seed})")


def _predict_with_independent_start(runner, predictive_class) -> None:
    """Official inference loop with one added ``initial_gaze`` model argument."""

    import torch
    from suppor_lib import image_process  # type: ignore

    rep_num = int(runner.n_scanpaths)
    predictive = predictive_class(runner.dmm.model, num_samples=1)
    for image_path in sorted(Path(runner.img_path).glob("*")):
        image_tensor = torch.unsqueeze(image_process(str(image_path)), dim=0).repeat(
            [rep_num, 1, 1, 1]
        )
        initial_gaze = runner.create_random_starting_points(rep_num).to(torch.float32)
        dummy_scanpaths = initial_gaze.unsqueeze(1).repeat([1, runner.length, 1])
        mask = torch.ones([rep_num, runner.length])
        with torch.no_grad():
            samples = predictive(
                scanpaths=dummy_scanpaths.cuda(),
                scanpaths_reversed=None,
                mask=mask.cuda(),
                scanpath_lengths=None,
                images=image_tensor.cuda(),
                predict=True,
                initial_gaze=initial_gaze.cuda(),
            )
        scanpaths = runner.summary(samples).cpu().numpy()
        output = Path(runner.output_path)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / f"{image_path.stem}.npy", scanpaths)
        print(f"{image_path.name}: {scanpaths.shape[0]} scanpaths -> {output}", flush=True)


if __name__ == "__main__":
    main()
