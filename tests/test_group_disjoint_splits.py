from __future__ import annotations

import json

import pytest

from sphererate.data.splits import (
    apply_group_disjoint_splits,
    audit_conditioned_pairs,
    audit_group_disjoint_splits,
    group_id_for_row,
)


def test_group_id_for_known_multi360_datasets() -> None:
    assert group_id_for_row({"dataset": "jufe", "source_image_id": "100_len5_st_3"}) == "jufe:100"
    assert (
        group_id_for_row({"dataset": "sitzmann", "source_image_id": "cubemap_0001_4"})
        == "sitzmann:cubemap_0001"
    )
    assert (
        group_id_for_row({"dataset": "sitzmann", "source_image_id": "cubemap_0000"})
        == "sitzmann:cubemap_0000"
    )
    assert (
        group_id_for_row({"dataset": "sitzmann", "source_image_id": "cubemap_0006"})
        == "sitzmann:cubemap_0006"
    )
    assert group_id_for_row({"dataset": "salient360", "image_id": "P17"}) == "salient360:P17"


def test_explicit_canonical_content_unifies_etdk2_and_salient360_aliases() -> None:
    salient = {
        "dataset": "salient360",
        "image_id": "P7",
        "canonical_content_id": "salient360:P7",
    }
    etdk2 = {
        "dataset": "etdk2",
        "image_id": "etdk2_P7_S0002_t01",
        "source_image_id": "P7",
        "canonical_content_id": "salient360:P7",
    }

    assert group_id_for_row(salient) == "content:salient360:P7"
    assert group_id_for_row(etdk2) == "content:salient360:P7"

    assigned, _, audit = apply_group_disjoint_splits(
        [salient, etdk2], val_fraction=0.0, test_fraction=0.0, seed=7
    )
    assert {row["split"] for row in assigned} == {"train"}
    assert audit["zero_leakage"] is True


def test_group_id_rejects_malformed_known_dataset_id() -> None:
    with pytest.raises(ValueError, match="JUFE"):
        group_id_for_row({"dataset": "jufe", "source_image_id": "not-a-jufe-id"})


def test_apply_group_disjoint_splits_is_deterministic_and_keeps_groups_together() -> None:
    rows = []
    for content in range(12):
        for variant in range(2):
            rows.append(
                {
                    "dataset": "jufe",
                    "image_id": f"jufe_{content}_len1_st_{variant}",
                    "source_image_id": f"{content}_len1_st_{variant}",
                    "split": "train" if variant == 0 else "test",
                }
            )

    assigned_a, spec_a, audit_a = apply_group_disjoint_splits(
        rows,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )
    assigned_b, spec_b, audit_b = apply_group_disjoint_splits(
        rows,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )

    assert spec_a == spec_b
    assert [row["split"] for row in assigned_a] == [row["split"] for row in assigned_b]
    assert audit_a == audit_b
    assert audit_a["leaking_group_count"] == 0
    assert set(audit_a["per_dataset"]["jufe"]["groups_by_split"].values()) == {3, 6}

    split_by_group: dict[str, set[str]] = {}
    for row in assigned_a:
        split_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_group.values())


def test_audit_group_disjoint_splits_reports_leakage() -> None:
    rows = [
        {
            "dataset": "sitzmann",
            "image_id": "sitzmann_cubemap_0001_0",
            "source_image_id": "cubemap_0001_0",
            "split": "train",
        },
        {
            "dataset": "sitzmann",
            "image_id": "sitzmann_cubemap_0001_1",
            "source_image_id": "cubemap_0001_1",
            "split": "test",
        },
    ]

    audit = audit_group_disjoint_splits(rows)

    assert audit["leaking_group_count"] == 1
    assert audit["leaking_groups"][0]["group_id"] == "sitzmann:cubemap_0001"
    assert audit["leaking_groups"][0]["splits"] == ["test", "train"]


def test_apply_group_disjoint_splits_spec_is_json_serializable() -> None:
    rows = [
        {"dataset": "salient360", "image_id": f"P{idx}", "split": "train"}
        for idx in range(1, 8)
    ]

    _, spec, audit = apply_group_disjoint_splits(rows, 0.2, 0.2, seed=3)

    json.dumps(spec)
    json.dumps(audit)


def test_audit_conditioned_pairs_requires_good_bad_same_split_and_group() -> None:
    rows = [
        {
            "dataset": "jufe_start_conditioned",
            "image_id": f"jufe_100_len1_st_1__start_{condition}",
            "base_image_id": "jufe_100_len1_st_1",
            "source_image_id": "100_len1_st_1",
            "group_id": "jufe:100",
            "split": "val",
            "condition": condition,
        }
        for condition in ("good", "bad")
    ]

    audit = audit_conditioned_pairs(rows)

    assert audit["zero_leakage"] is True
    assert audit["pair_complete"] is True
    assert audit["base_image_count"] == 1
    assert audit["incomplete_pair_count"] == 0

    rows[1]["split"] = "test"
    broken = audit_conditioned_pairs(rows)
    assert broken["zero_leakage"] is False
    assert broken["pair_complete"] is False
    assert broken["pair_split_mismatch_count"] == 1
