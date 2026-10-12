# Workspace Cleanup — 2026-07-13

This cleanup keeps the kernel-v2 ICASSP evidence chain reproducible while removing generated artifacts from closed or invalidated branches.

## Kept

- `data/processed/early_horizon_salient360_spherical2deg_v2`: current spherical-2° targets and train-only templates.
- `data/processed/salient360_256x512_k6_order`: source images referenced by the v2 manifest.
- `data/processed/etdk2_salient360shared_256x512_k6_session_v1`: current external multi-start diagnostic data.
- `data/processed/kernel_v2_matched_scanpaths`: matched scanpath evidence.
- `data/sitzmann_raw_data.zip`: official raw Sitzmann archive needed for a correct re-ingestion; integrity-tested, 4,559 entries, with 1,800 trial CSVs carrying viewpoint suffixes 1–4. SHA-256: `26b221416e85ccc03cdecc033e54120da953968a8a9c54a6aec09bd155a4df9f`.
- `data/interim/sitzmann_official_vr_v1` and `data/processed/sitzmann_official_256x512_spherical2deg_v1`: rebuilt after cleanup from the official standing-VR pickle family with trial-level `viewpoint_idx` retained.
- kernel-v2 SphereHorizon, attribution-control, Panoramic-CNN, TempSAL, DeepGaze, Pathformer3D, reachable-final, and corrected declared-start ScanDMM runs.
- reports, configs, source code, and small historical summaries needed to understand closed decisions.

## Removed generated data

- all JUFE processed roots and smoke views;
- obsolete `Sitzmann + Salient360 + JUFE` Multi360 manifest views;
- closed P2 processed view;
- the local ScanDMM-derived Sitzmann processed root;
- the combined `future_saliency_etdk2_sitzmann_salient360_v1` view;
- R0's 1.3 GB prediction cache, while retaining its summary/results;
- temporary files and Python/test caches.

JUFE was removed because it is historical-diagnostic only. The local Sitzmann data were removed because they lack the official per-trial `viewpoint_idx`, so they cannot support a valid start-aware transfer test.

## Removed runs

Removed 171 run directories (about 51.9 GiB): old final-map/F2, Multi360, JUFE, R1/P1/P2/Gate A–C, v1 EH, smoke runs, and the superseded first-target-initialized ScanDMM runs. Numeric reports remain, but their deleted checkpoints/prediction caches must be regenerated from the retained code/configs if a historical audit is explicitly reopened.

Within the retained Pathformer3D and declared-start ScanDMM runs, 81 unselected epoch checkpoints (about 2.14 GiB) were also removed. Each seed's val-selected checkpoint, `selection.json`, training log, hashes, and val/gate inference artifacts remain.

## Sitzmann re-entry rule — fulfilled on 2026-07-13

Do not recreate the deleted root from the current ScanDMM pickle. A new Sitzmann view must originate from official trial-level records and retain:

- scene identity;
- subject/trial identity;
- `viewpoint_idx ∈ {0,1,2,3}`;
- the verified mapping from viewpoint index to ERP yaw;
- timestamps and one-sample-per-second semantics;
- content-disjoint grouping of every derived rotation.

The new official view satisfies these requirements: 22 scenes, four viewpoints per scene, 780 trials, 774 cell-level physical observers after merging six repeated-subject trials, and a zero-overlap 16/3/3 scene split. It is authorized as multi-start external evidence and secondary adaptation, but the controlled training-source comparison does not support replacing Salient360 as the primary training set.
