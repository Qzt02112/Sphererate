from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_EVALUATION_SPLITS = {"val", "gate", "test"}
ALLOWED_TRAINING_POLICIES = {
    "same_split",
    "train_only_calibration",
    "not_applicable",
    "external_pretrained",
}
PRIMARY_TRAINING_POLICIES = {"same_split", "train_only_calibration", "not_applicable"}
ALLOWED_STATUSES = {
    "planned",
    "smoke_pass",
    "trained",
    "evaluated",
    "evaluated_with_training_condition_caveat",
    "diagnostic_only",
    "ineligible_hash_image_condition",
    "blocked",
}


@dataclass(frozen=True)
class MatchedMethod:
    name: str
    tier: str
    task: str
    training_policy: str
    source: str
    source_commit: str
    status: str


@dataclass(frozen=True)
class MatchedBaselineProtocol:
    name: str
    data_root: Path
    manifest: Path
    train_split: str
    selection_split: str
    evaluation_splits: tuple[str, ...]
    sealed_splits: tuple[str, ...]
    expected_content_counts: dict[str, int]
    primary_eye: str
    horizons_s: tuple[float, ...]
    scanpath_step_s: float
    scanpath_length: int
    half_second_policy: str
    start_pitch_deg: float
    start_yaw_deg: float
    seeds: tuple[int, ...]
    primary_method_names: tuple[str, ...]
    methods: dict[str, MatchedMethod]
    map_adapter: dict[str, Any]
    start_mode: str
    pass_start_to_conditionable_models: bool
    reachability: dict[str, Any]

    @property
    def primary_methods(self) -> tuple[MatchedMethod, ...]:
        return tuple(self.methods[name] for name in self.primary_method_names)


def load_matched_protocol(path: str | Path) -> MatchedBaselineProtocol:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Matched baseline protocol must be a mapping")
    data = _mapping(raw, "data")
    time = _mapping(raw, "time")
    start = _mapping(raw, "start")
    adapter = _mapping(raw, "map_adapter")
    method_rows = _mapping(raw, "methods")
    reachability = _mapping(raw, "reachability")
    methods = {
        str(name): MatchedMethod(
            name=str(name),
            tier=str(_required(row, "tier", f"methods.{name}")),
            task=str(_required(row, "task", f"methods.{name}")),
            training_policy=str(_required(row, "training_policy", f"methods.{name}")),
            source=str(_required(row, "source", f"methods.{name}")),
            source_commit=str(_required(row, "source_commit", f"methods.{name}")),
            status=str(_required(row, "status", f"methods.{name}")),
        )
        for name, row in method_rows.items()
    }
    protocol = MatchedBaselineProtocol(
        name=str(_required(raw, "protocol", "root")),
        data_root=Path(str(_required(data, "root", "data"))),
        manifest=Path(str(_required(data, "manifest", "data"))),
        train_split=str(_required(data, "train_split", "data")),
        selection_split=str(_required(data, "selection_split", "data")),
        evaluation_splits=tuple(str(value) for value in _required(data, "evaluation_splits", "data")),
        sealed_splits=tuple(str(value) for value in _required(data, "sealed_splits", "data")),
        expected_content_counts={
            str(key): int(value) for key, value in _mapping(data, "expected_content_counts").items()
        },
        primary_eye=str(_required(data, "primary_eye", "data")),
        horizons_s=tuple(float(value) for value in _required(time, "horizons_s", "time")),
        scanpath_step_s=float(_required(time, "scanpath_step_s", "time")),
        scanpath_length=int(_required(time, "scanpath_length", "time")),
        half_second_policy=str(_required(time, "half_second_policy", "time")),
        start_pitch_deg=float(_required(start, "pitch_deg", "start")),
        start_yaw_deg=float(_required(start, "yaw_deg", "start")),
        seeds=tuple(int(value) for value in _required(raw, "seeds", "root")),
        primary_method_names=tuple(str(value) for value in _required(raw, "primary_methods", "root")),
        methods=methods,
        map_adapter=dict(adapter),
        start_mode=str(_required(start, "mode", "start")),
        pass_start_to_conditionable_models=bool(
            _required(start, "pass_to_conditionable_models", "start")
        ),
        reachability=dict(reachability),
    )
    _validate(protocol)
    return protocol


def _validate(protocol: MatchedBaselineProtocol) -> None:
    if not protocol.evaluation_splits or not set(protocol.evaluation_splits) <= ALLOWED_EVALUATION_SPLITS:
        raise ValueError(f"Evaluation splits must be a non-empty subset of {sorted(ALLOWED_EVALUATION_SPLITS)}")
    if set(protocol.evaluation_splits) & set(protocol.sealed_splits):
        raise ValueError("Evaluation and sealed splits overlap")
    standard = "standard_split" in protocol.name
    if standard:
        if protocol.sealed_splits:
            raise ValueError("The standard protocol does not seal evaluation splits")
        if "test" not in protocol.evaluation_splits:
            raise ValueError("The standard protocol must report the test split")
    elif "test" not in protocol.sealed_splits:
        raise ValueError("The historical v2 test split must remain sealed")
    if protocol.train_split != "train" or protocol.selection_split != "val":
        raise ValueError("Matched protocol requires train-only fitting and val-only selection")
    if protocol.primary_eye != "R":
        raise ValueError("Matched protocol must use the frozen right-eye primary analysis")
    if protocol.horizons_s != (0.5, 1.0, 3.0, 5.0):
        raise ValueError(f"Unexpected horizons: {protocol.horizons_s}")
    if protocol.scanpath_step_s != 1.0 or protocol.scanpath_length < int(max(protocol.horizons_s)):
        raise ValueError("Scanpath time semantics do not cover the frozen horizons")
    if len(protocol.seeds) != 3 or len(set(protocol.seeds)) != 3:
        raise ValueError("Exactly three distinct seeds are required")
    if set(protocol.primary_method_names) - set(protocol.methods):
        raise ValueError("A primary method is missing from the method registry")
    for method in protocol.methods.values():
        if method.training_policy not in ALLOWED_TRAINING_POLICIES:
            raise ValueError(f"Unknown training policy for {method.name}: {method.training_policy}")
        if method.status not in ALLOWED_STATUSES:
            raise ValueError(f"Unknown status for {method.name}: {method.status}")
        if not method.source or not method.source_commit:
            raise ValueError(f"Missing source provenance for {method.name}")
    for method in protocol.primary_methods:
        if method.training_policy not in PRIMARY_TRAINING_POLICIES:
            raise ValueError(f"External pretrained method cannot enter the primary table: {method.name}")
        if method.tier in {"external_pretrained", "local_reproduction"}:
            raise ValueError(f"Ineligible tier in primary table: {method.name} ({method.tier})")
    if bool(protocol.map_adapter.get("display_smoothing_affects_metrics", True)):
        raise ValueError("Display smoothing must never affect metrics")
    if protocol.start_mode != "per_sample_declared_input" or not protocol.pass_start_to_conditionable_models:
        raise ValueError("Conditionable baselines must receive the declared start as a model input")
    if str(protocol.reachability.get("calibration_split")) != "train":
        raise ValueError("Reachability radii must be calibrated on train only")
    if float(protocol.reachability.get("quantile", -1.0)) != 0.95:
        raise ValueError("The frozen reachability quantile must remain 0.95")
    expected_radii = {0.5, 1.0, 3.0, 5.0}
    radii = {float(key): float(value) for key, value in _mapping(protocol.reachability, "radii_deg").items()}
    if set(radii) != expected_radii or any(radii[h] <= 0.0 for h in expected_radii):
        raise ValueError("Reachability radii must cover every frozen horizon")
    if any(radii[a] > radii[b] for a, b in zip(sorted(radii)[:-1], sorted(radii)[1:])):
        raise ValueError("Reachability radii must be non-decreasing")


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(parent, key, "root")
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required(parent: Any, key: str, scope: str) -> Any:
    if not isinstance(parent, dict) or key not in parent:
        raise ValueError(f"Missing required field {scope}.{key}")
    return parent[key]
