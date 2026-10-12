#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

test -f RESEARCH_PROTOCOL.md
test -f Goal.md
test -f data/sitzmann_raw_data.zip
test -f reports/sitzmann_cross_dataset_benchmark/results.json

keep_only() {
  local directory="$1"
  local keep_name="$2"
  test -d "$directory" || return 0
  find "$directory" -maxdepth 1 -type f ! -name "$keep_name" -delete
}

# Rebuildable environments, archives, caches, and logs.
rm -rf archive/off_mainline_nextfix_20260710 third_party/.envs
rm -rf .pytest_cache tmp logs Log
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# Superseded processed datasets. The spherical-2deg v2 and 256x512 sources remain.
rm -rf data/processed/early_horizon_salient360_v1
rm -rf data/processed/salient360_128x256_k6_order

# Keep only validation-selected periodic checkpoints for the expensive baselines.
keep_only runs/kernel_v2_hat_seed20260710_train/checkpoints step_004000.pt
keep_only runs/kernel_v2_hat_seed20260711_train/checkpoints step_005000.pt
keep_only runs/kernel_v2_hat_seed20260712_train/checkpoints step_004000.pt

for seed in 20260710 20260711 20260712; do
  keep_only "runs/sitzmann_only_pathformer3d_seed${seed}/checkpoints" epoch_0040.pt
done
keep_only runs/sitzmann_only_scandmm_declared_start_seed20260710/checkpoints epoch_0399.pkl
keep_only runs/sitzmann_only_scandmm_declared_start_seed20260711/checkpoints epoch_0499.pkl
keep_only runs/sitzmann_only_scandmm_declared_start_seed20260712/checkpoints epoch_0499.pkl

# Closed controls and smoke runs already have frozen report evidence.
rm -rf runs/kernel_v2_hat_smoke_seed20260710*
rm -rf runs/semantic_reranking_s1_seed20260711_smoke
rm -rf runs/semantic_reranking_s2_seed20260711_smoke
rm -rf runs/semantic_reranking_s2_matched_anchor_seed20260711_smoke
rm -rf runs/kernel_v2_candidate_shuffled_label_seed20260711
rm -rf runs/kernel_v2_capacity_non_temporal_seed20260711
rm -rf runs/kernel_v2_static_final_seed20260711

# Prediction arrays and duplicate last checkpoints are deterministic products.
find runs -mindepth 2 -maxdepth 2 -type d -name predictions -prune -exec rm -rf {} +
while IFS= read -r checkpoint_dir; do
  if test -f "$checkpoint_dir/best.pt"; then
    rm -f "$checkpoint_dir/last.pt"
  fi
done < <(find runs -mindepth 2 -maxdepth 2 -type d -name checkpoints)

# Old task routes and superseded qualitative caches.
rm -rf \
  reports/content_utilization_audit \
  reports/eh5_outer_test reports/eh5_outer_test_access \
  reports/eh6_benchmark reports/eh6_panoramic_static_test reports/eh6_panoramic_static_val \
  reports/eh6_saltinet_pretrained_test reports/eh6_saltinet_pretrained_val \
  reports/eh6_scandmm_declaredstart_test reports/eh6_scandmm_declaredstart_val \
  reports/eh6_scandmm_test reports/eh6_scandmm_val reports/eh6_visualizations \
  reports/f0_human_upper_bound reports/f0_priors reports/f0_visuals reports/f1_quick \
  reports/f2_ablation reports/f2_optimization_256 reports/f2_optimization_smoke_256 \
  reports/f2_process_yaw60_256 reports/f2_residual_mixer_256 reports/f2_semantic_2x2_256 \
  reports/f2_teacher_residual_256 reports/f2_tempsal_staged_256 reports/f2_visualizations \
  reports/f2_yaw60_control_256 reports/f3_scandmm_200 \
  reports/gate_a_static_clean reports/gate_a_static_clean_pilot \
  reports/gate_b_condition_signal reports/gate_c_marginalization \
  reports/gate_eh2_data_audit reports/gate_eh3_template_residual reports/gate_eh4_etdk2 \
  reports/gate_eh4_robustness reports/gate_eh6_external_baselines \
  reports/gate_p2_temporal_aux reports/gate_r0_content_interaction \
  reports/gate_r1_start_conditioned_late reports/gate_t0_temporal_decomposition \
  reports/jufe_start_conditioned_smoke reports/jufe_start_salmap_similarity \
  reports/mainline_visualizations reports/matched_static_semantic_anchor \
  reports/multi360_frame_align_256 reports/multi360_prefix_conditioned_probe_256 \
  reports/multi360_prefix_conditioned_smoke reports/multi360_process_task_256 \
  reports/multi360_semantic_2x2_256 reports/multi360_smoke reports/multi360_temporal_hard_256 \
  reports/p1_condition_audit reports/p1_conditioning_pilot reports/protocol reports/smoke_eval

# Selection sweeps: keep only the three chosen declared-start checkpoints/reports.
find reports -mindepth 1 -maxdepth 1 -type d -name 'kernel_v2_scandmm_selection_seed*' -exec rm -rf {} +
find reports -mindepth 1 -maxdepth 1 -type d -name 'kernel_v2_scandmm_declared_start_selection_seed*' \
  ! -name 'kernel_v2_scandmm_declared_start_selection_seed20260710_epoch0399' \
  ! -name 'kernel_v2_scandmm_declared_start_selection_seed20260711_epoch0374' \
  ! -name 'kernel_v2_scandmm_declared_start_selection_seed20260712_epoch0349' -exec rm -rf {} +
find reports -mindepth 1 -maxdepth 1 -type d -name 'sitzmann_only_scandmm_declared_start_selection_seed*' \
  ! -name 'sitzmann_only_scandmm_declared_start_selection_seed20260710_epoch0399' \
  ! -name 'sitzmann_only_scandmm_declared_start_selection_seed20260711_epoch0499' \
  ! -name 'sitzmann_only_scandmm_declared_start_selection_seed20260712_epoch0499' -exec rm -rf {} +

# Obsolete root-level planning artifact; current decisions live in Goal/Protocol/handoff.
rm -f 'DyTSalDMM360_research_plan(1).md'

echo "cleanup complete"
