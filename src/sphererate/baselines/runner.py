from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class _FormatVars(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class BaselineTask:
    name: str
    stage: str
    command: str
    out: str
    exists: bool
    status: str


def _parse_vars(items: list[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}")
        key, value = item.split("=", 1)
        values[key] = value
    return values


def _default_out(name: str, stage: str, dataset: str, split: str) -> str:
    suffix = split if stage != "train" else dataset
    return f"runs/baselines/{name}_{stage}_{suffix}"


def build_baseline_task(
    registry: dict[str, Any],
    name: str,
    stage: str = "train",
    dataset: str = "salient360",
    split: str = "val",
    out: str | None = None,
    device: str = "cuda",
    experiment_config: str = "configs/default.yaml",
    extra_vars: dict[str, str] | None = None,
) -> BaselineTask:
    stage = stage.lower()
    out = out or _default_out(name, stage, dataset, split)
    variables = _FormatVars(
        {
            "name": name,
            "stage": stage,
            "dataset": dataset,
            "split": split,
            "out": out,
            "device": device,
            "config": experiment_config,
            "n_scanpaths": "200",
            "length": "20",
            "seed": "1234",
        }
    )
    variables.update(extra_vars or {})

    if name in registry.get("ours", {}):
        item = registry["ours"][name]
        template = item.get(f"{stage}_command") or item.get("command")
        if not template:
            raise KeyError(f"No command registered for internal baseline {name!r} stage {stage!r}")
        return BaselineTask(
            name=name,
            stage=stage,
            command=template.format_map(variables),
            out=out,
            exists=True,
            status=str(item.get("category", "internal")),
        )

    item = registry.get("baselines", {}).get(name)
    if not item:
        raise KeyError(f"Unknown baseline: {name}")
    template = item.get(f"{stage}_hint") or item.get("command")
    if not template:
        raise KeyError(f"No {stage!r} command hint registered for baseline {name!r}")
    path = item.get("resolved_path") or item.get("local_path") or ""
    variables["path"] = str(path)
    return BaselineTask(
        name=name,
        stage=stage,
        command=str(template).format_map(variables),
        out=out,
        exists=bool(item.get("exists")),
        status=str(item.get("status", "")),
    )


def run_task(task: BaselineTask, dry_run: bool = False) -> int:
    print(task.command)
    if dry_run:
        return 0
    Path(task.out).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(task.command, shell=True, check=False)
    return int(completed.returncode)


def load_batch_tasks(
    registry: dict[str, Any],
    batch_config: str | Path,
    default_device: str = "cuda",
    default_experiment_config: str = "configs/default.yaml",
) -> list[BaselineTask]:
    with open(batch_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    defaults = cfg.get("defaults", {})
    tasks: list[BaselineTask] = []
    for item in cfg.get("tasks", []):
        extra = {str(k): str(v) for k, v in (item.get("vars") or {}).items()}
        tasks.append(
            build_baseline_task(
                registry,
                name=str(item["name"]),
                stage=str(item.get("stage", defaults.get("stage", "train"))),
                dataset=str(item.get("dataset", defaults.get("dataset", "salient360"))),
                split=str(item.get("split", defaults.get("split", "val"))),
                out=item.get("out"),
                device=str(item.get("device", defaults.get("device", default_device))),
                experiment_config=str(item.get("config", defaults.get("config", default_experiment_config))),
                extra_vars=extra,
            )
        )
    return tasks


def parse_cli_vars(items: list[str] | None) -> dict[str, str]:
    return _parse_vars(items)
