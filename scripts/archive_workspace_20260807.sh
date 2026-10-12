#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCHIVE="temp/workspace_cleanup_20260807"
if test -f "$ARCHIVE/_ARCHIVE_README.md"; then
  echo "Archive already finalized at $ARCHIVE"
  exit 0
fi
mkdir -p "$ARCHIVE"

archive_path() {
  local source="$1"
  test -e "$source" || return 0
  local destination="$ARCHIVE/$source"
  test ! -e "$destination"
  mkdir -p "$(dirname "$destination")"
  mv "$source" "$destination"
}

# Root caches and transient render directories.
archive_path .pytest_cache
archive_path Log
archive_path tmp
mkdir -p tmp

# Rebuildable baseline environments and Python bytecode are not source assets.
archive_path third_party/.envs
while IFS= read -r cache_dir; do
  archive_path "$cache_dir"
done < <(find src scripts tests configs paper third_party -type d -name __pycache__ -prune | sort)

# Tests whose contracts explicitly require the archived V2/kernel-v2/cross-
# dataset artifacts move with those historical routes. Current V3 tests remain.
for path in \
  tests/test_early_horizon_dataset.py \
  tests/test_kernel_v2_paper_artifacts.py \
  tests/test_matched_baseline_protocol.py \
  tests/test_sitzmann_cross_dataset_contract.py; do
  archive_path "$path"
done

# LaTeX products are reproducible; source, bibliography, tables, figures, and
# the latest paper/main.pdf remain active.
for path in \
  paper/main.aux paper/main.blg paper/main.fdb_latexmk paper/main.fls \
  paper/main.log paper/main.out paper/tmp; do
  archive_path "$path"
done

# Superseded paper assets that are not included by paper/main.tex.
for path in \
  paper/figures/codec_application_vertical_tikz.tex \
  paper/figures/horizon_behavior.pdf \
  paper/figures/matched_gate_benchmark.pdf \
  paper/figures/matched_qualitative_3s.pdf \
  paper/figures/matched_qualitative_3s_compact.pdf \
  paper/figures/matched_tile_decisions_3s.pdf \
  paper/figures/spherehorizon_overview.pdf \
  paper/tables/table_ablation_deltas.tex \
  paper/tables/table_ablation_horizon.tex \
  paper/tables/table_codec_application.tex \
  paper/tables/table_efficiency.tex \
  paper/tables/table_main_gate.tex \
  paper/tables/table_scandmm_correction.tex \
  paper/tables/table_sitzmann_transfer.tex \
  paper/tables/table_sphererate_ablation.tex; do
  archive_path "$path"
done

# Old preview/draft names; retain the current SphereHorizon PDF and editable PPTX.
archive_path output/pdf/SphereRate_ICASSP_draft.pdf
archive_path output/preview

# Superseded data views. The active V3 Salient360 view, native Salient360
# source, codec cache, and Sitzmann standard-v2 view stay in data/.
for path in \
  data/processed/cross_scanpaths_salient360_at_sitzmann_start_v1 \
  data/processed/cross_scanpaths_sitzmann_at_salient360_start_v1 \
  data/processed/cross_view_salient360_at_sitzmann_start_v1 \
  data/processed/cross_view_sitzmann_at_salient360_start_v1 \
  data/processed/early_horizon_salient360_spherical2deg_v2 \
  data/processed/etdk2_salient360shared_256x512_k6_session_v1 \
  data/processed/kernel_v2_matched_scanpaths \
  data/processed/sitzmann_official_256x512_spherical2deg_v1 \
  data/processed/sitzmann_official_matched_scanpaths_v1 \
  data/processed/v11_raw_tile_consensus_seed20260805; do
  archive_path "$path"
done

# Only the current standard-split and Sitzmann same-domain run families remain
# under runs/. All earlier routes are preserved in the archive.
for path in runs/*; do
  test -e "$path" || continue
  case "$(basename "$path")" in
    standard_split_v3|sitzmann_standard_v2) ;;
    *) archive_path "$path" ;;
  esac
done

# Remove smoke/interrupted runs from the active run tree.
for path in \
  runs/standard_split_v3/hat_train_seed20260710_aborted_contention \
  runs/standard_split_v3/scandmm_train_seed20260710_aborted_contention \
  runs/standard_split_v3/scandmm_train_seed20260711_aborted_contention \
  runs/standard_split_v3/scandmm_train_seed20260712_aborted_contention \
  runs/standard_split_v3/logs \
  runs/standard_split_v3/v12c_no_rate_seed20260711_smoke \
  runs/standard_split_v3/v12c_no_start_seed20260711_smoke \
  runs/standard_split_v3/v12c_no_tile_loss_seed20260711_smoke \
  runs/sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260711_smoke \
  runs/sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260711_salient_scale_diagnostic; do
  archive_path "$path"
done

# Keep validation-selected checkpoints and the final checkpoint required by the
# protocol-completeness audit; archive the remaining periodic snapshots.
for seed in 20260710 20260711 20260712; do
  for checkpoint in runs/standard_split_v3/pathformer3d_train_seed${seed}/checkpoints/*; do
    test -e "$checkpoint" || continue
    name="$(basename "$checkpoint")"
    case "${seed}:${name}" in
      20260710:epoch_0035.pt|20260710:epoch_0040.pt|\
      20260711:epoch_0040.pt|20260712:epoch_0040.pt) ;;
      *) archive_path "$checkpoint" ;;
    esac
  done

  for checkpoint in runs/standard_split_v3/scandmm_train_seed${seed}/checkpoints/*; do
    test -e "$checkpoint" || continue
    name="$(basename "$checkpoint")"
    case "${seed}:${name}" in
      20260710:epoch_0399.pkl|20260710:epoch_0499.pkl|\
      20260711:epoch_0424.pkl|20260711:epoch_0499.pkl|\
      20260712:epoch_0324.pkl|20260712:epoch_0499.pkl) ;;
      *) archive_path "$checkpoint" ;;
    esac
  done

  for checkpoint in runs/standard_split_v3/hat_train_seed${seed}/checkpoints/*; do
    test -e "$checkpoint" || continue
    name="$(basename "$checkpoint")"
    case "${seed}:${name}" in
      20260710:step_003000.pt|20260710:step_005000.pt|\
      20260711:step_003000.pt|20260711:step_005000.pt|\
      20260712:step_005000.pt) ;;
      *) archive_path "$checkpoint" ;;
    esac
  done
done

# Keep only selected ScanDMM validation inference runs.
for seed in 20260710 20260711 20260712; do
  selected=""
  case "$seed" in
    20260710) selected="epoch_0399_n50" ;;
    20260711) selected="epoch_0424_n50" ;;
    20260712) selected="epoch_0324_n50" ;;
  esac
  for path in runs/standard_split_v3/scandmm_selection_seed${seed}/*; do
    test -d "$path" || continue
    test "$(basename "$path")" = "$selected" || archive_path "$path"
  done
done

# When a validation-selected best.pt exists, last.pt is a redundant training
# continuation snapshot for the current deliverable.
while IFS= read -r checkpoint_dir; do
  if test -f "$checkpoint_dir/best.pt" && test -f "$checkpoint_dir/last.pt"; then
    archive_path "$checkpoint_dir/last.pt"
  fi
done < <(find runs/standard_split_v3 runs/sitzmann_standard_v2 -type d -name checkpoints | sort)

# Keep current standard-split/SphereRate/codec/Sitzmann-same-domain evidence and
# compact source audits; archive superseded experiment report directories.
for path in reports/*; do
  test -d "$path" || continue
  name="$(basename "$path")"
  case "$name" in
    standard_split_*|sphererate_*|codec_tile_prefetch_test|sitzmann_standard_v2|\
    saliency_kernel_audit|sitzmann_official_rebuild|v12c_spherical_dwell_rate|\
    bms360_gbvs360_source_audit|pathformer3d_matched_audit|\
    recent_saliency_2025_2026_source_audit|saldiff_dtm_source_audit|\
    scandmm_declared_start_semantics_audit|scandmm_training_semantics_comparison) ;;
    *) archive_path "$path" ;;
  esac
done

# Standard ScanDMM selection sweep reports: retain only the three selected rows.
for path in reports/standard_split_scandmm_selection_selection_seed*; do
  test -d "$path" || continue
  name="$(basename "$path")"
  case "$name" in
    standard_split_scandmm_selection_selection_seed20260710_epoch0399|\
    standard_split_scandmm_selection_selection_seed20260711_epoch0424|\
    standard_split_scandmm_selection_selection_seed20260712_epoch0324) ;;
    *) archive_path "$path" ;;
  esac
done

# Historical implementation plans are no longer active specifications.
archive_path docs/plans

echo "Archived non-mainline workspace material under $ARCHIVE"
