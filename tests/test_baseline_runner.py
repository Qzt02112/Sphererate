from __future__ import annotations

from pathlib import Path

import pytest

from sphererate.baselines.registry import load_baseline_registry
from sphererate.baselines.runner import build_baseline_task, load_batch_tasks, parse_cli_vars


def _local_sources_present() -> bool:
    root = load_baseline_registry("configs/baselines.yaml").get("third_party_root", "")
    return bool(root) and Path(root).is_dir()


requires_local_sources = pytest.mark.skipif(
    not _local_sources_present(),
    reason="third-party baseline sources not checked out; set SCANDMM_ROOT / TEMPSAL_ROOT to enable",
)


@requires_local_sources
def test_internal_baseline_command_uses_templates() -> None:
    registry = load_baseline_registry("configs/baselines.yaml")
    task = build_baseline_task(registry, "static", device="cuda", experiment_config="configs/default.yaml")
    assert task.exists
    assert "--model static" in task.command
    assert "--device cuda" in task.command


@requires_local_sources
def test_scandmm_infer_command_can_be_parameterized() -> None:
    registry = load_baseline_registry("configs/baselines.yaml")
    task = build_baseline_task(
        registry,
        "scandmm",
        stage="infer",
        split="test",
        out="runs/scandmm_test_50",
        extra_vars=parse_cli_vars(["n_scanpaths=50", "length=12", "seed=7"]),
    )
    assert task.exists
    assert "--split test" in task.command
    assert "--run-dir runs/scandmm_test_50" in task.command
    assert "--n-scanpaths 50" in task.command
    assert "--length 12" in task.command
    assert "--seed 7" in task.command


@requires_local_sources
def test_baseline_batch_dryrun_config_loads() -> None:
    registry = load_baseline_registry("configs/baselines.yaml")
    tasks = load_batch_tasks(registry, Path("configs/baseline_batch_dryrun.yaml"))
    assert [task.name for task in tasks] == ["static", "parallel", "dytsaldmm", "scandmm"]
    assert tasks[-1].stage == "infer"
