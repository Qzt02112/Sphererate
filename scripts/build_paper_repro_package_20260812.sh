#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="SphereRate_ICASSP_reproduction_20260812"
OUT_ROOT="${ROOT}/output/packages"
STAGE="${OUT_ROOT}/${NAME}"
ARCHIVE="${OUT_ROOT}/${NAME}.tar.zst"
EXTERNAL_BASELINES="${SCANDMM_ROOT}/baselines/third_party"

command -v rsync >/dev/null
command -v zstd >/dev/null

rm -rf "${STAGE}"
mkdir -p "${STAGE}" "${OUT_ROOT}"

copy_tree() {
  local source="$1"
  local target="$2"
  mkdir -p "${target}"
  rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "${source}/" "${target}/"
}

copy_file() {
  local source="$1"
  local target="$2"
  mkdir -p "$(dirname "${target}")"
  cp -a "${source}" "${target}"
}

# Project implementation and research record.
for directory in src scripts configs tests docs reports paper; do
  copy_tree "${ROOT}/${directory}" "${STAGE}/${directory}"
done

for file in README.md Goal.md RESEARCH_PROTOCOL.md MAINLINE_MODEL_VERSIONS.md pyproject.toml; do
  copy_file "${ROOT}/${file}" "${STAGE}/${file}"
done

copy_file "${ROOT}/output/pdf/SphereHorizon_ICASSP_draft.pdf" \
  "${STAGE}/output/pdf/SphereHorizon_ICASSP_draft.pdf"
copy_file "${ROOT}/output/pptx/SphereHorizon_editable_framework.pptx" \
  "${STAGE}/output/pptx/SphereHorizon_editable_framework.pptx"

# Fixed split definitions and lightweight manifests; bulk datasets are omitted.
copy_tree "${ROOT}/data/splits" "${STAGE}/data/splits"
declare -a DATA_MANIFESTS=(
  "processed/early_horizon_salient360_spherical2deg_standard_v3/manifest.jsonl"
  "processed/salient360_256x512_k6_order/manifest.jsonl"
  "processed/salient360_standard_train59_val13_test13_seed20260711/manifest.jsonl"
  "processed/standard_split_v3_matched_scanpaths/manifest.jsonl"
  "processed/sitzmann_official_256x512_spherical2deg_standard_v2/manifest.jsonl"
  "interim/salient360_codec_originals/manifest.jsonl"
)
for relative in "${DATA_MANIFESTS[@]}"; do
  if [[ -f "${ROOT}/data/${relative}" ]]; then
    safe_name="${relative//\//__}"
    copy_file "${ROOT}/data/${relative}" "${STAGE}/data/manifests/${safe_name}"
  fi
done

# Baseline sources already kept in this workspace.
copy_tree "${ROOT}/third_party" "${STAGE}/third_party"

# The three scanpath baselines lived in the neighboring ScanDMM workspace.
for baseline in ScanDMM Pathformer3D HAT; do
  if [[ ! -d "${EXTERNAL_BASELINES}/${baseline}" ]]; then
    echo "Missing external baseline: ${EXTERNAL_BASELINES}/${baseline}" >&2
    exit 1
  fi
  copy_tree "${EXTERNAL_BASELINES}/${baseline}" "${STAGE}/third_party/${baseline}"
done
rm -rf "${STAGE}/third_party/ScanDMM/Datasets"

# Copy lightweight metadata for every retained training run.
declare -a RUN_DIRS=(
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260710
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260711
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260712
  standard_split_v3/v12c_no_start_seed20260710
  standard_split_v3/v12c_no_start_seed20260711
  standard_split_v3/v12c_no_start_seed20260712
  standard_split_v3/v12c_no_rate_seed20260710
  standard_split_v3/v12c_no_rate_seed20260711
  standard_split_v3/v12c_no_rate_seed20260712
  standard_split_v3/v12c_no_tile_loss_seed20260710
  standard_split_v3/v12c_no_tile_loss_seed20260711
  standard_split_v3/v12c_no_tile_loss_seed20260712
  standard_split_v3/semantic_anchor_seed20260711
  standard_split_v3/static_anchor_seed20260711
  standard_split_v3/tempsal_erp_seed20260710
  standard_split_v3/tempsal_erp_seed20260711
  standard_split_v3/tempsal_erp_seed20260712
  standard_split_v3/panoramic_cnn_seed20260710
  standard_split_v3/panoramic_cnn_seed20260711
  standard_split_v3/panoramic_cnn_seed20260712
  standard_split_v3/pathformer3d_train_seed20260710
  standard_split_v3/pathformer3d_train_seed20260711
  standard_split_v3/pathformer3d_train_seed20260712
  standard_split_v3/scandmm_train_seed20260710
  standard_split_v3/scandmm_train_seed20260711
  standard_split_v3/scandmm_train_seed20260712
  standard_split_v3/hat_train_seed20260710
  standard_split_v3/hat_train_seed20260711
  standard_split_v3/hat_train_seed20260712
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260710
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260711
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260712
  sitzmann_standard_v2/semantic_anchor_seed20260711
  sitzmann_standard_v2/static_anchor_seed20260711
)
for relative in "${RUN_DIRS[@]}"; do
  mkdir -p "${STAGE}/runs/${relative}"
  rsync -a \
    --exclude='checkpoints/' \
    --exclude='*.npy' --exclude='*.npz' \
    --exclude='*.png' --exclude='*.jpg' --exclude='*.jpeg' \
    "${ROOT}/runs/${relative}/" "${STAGE}/runs/${relative}/"
done

# Validation-selected checkpoints used for the reported results.
declare -a CHECKPOINTS=(
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260710/checkpoints/best.pt
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260711/checkpoints/best.pt
  standard_split_v3/v12c_hybrid_dwell_rate_seed20260712/checkpoints/best.pt
  standard_split_v3/v12c_no_start_seed20260710/checkpoints/best.pt
  standard_split_v3/v12c_no_start_seed20260711/checkpoints/best.pt
  standard_split_v3/v12c_no_start_seed20260712/checkpoints/best.pt
  standard_split_v3/v12c_no_rate_seed20260710/checkpoints/best.pt
  standard_split_v3/v12c_no_rate_seed20260711/checkpoints/best.pt
  standard_split_v3/v12c_no_rate_seed20260712/checkpoints/best.pt
  standard_split_v3/v12c_no_tile_loss_seed20260710/checkpoints/best.pt
  standard_split_v3/v12c_no_tile_loss_seed20260711/checkpoints/best.pt
  standard_split_v3/v12c_no_tile_loss_seed20260712/checkpoints/best.pt
  standard_split_v3/semantic_anchor_seed20260711/checkpoints/best.pt
  standard_split_v3/static_anchor_seed20260711/checkpoints/best.pt
  standard_split_v3/tempsal_erp_seed20260710/checkpoints/best.pt
  standard_split_v3/tempsal_erp_seed20260711/checkpoints/best.pt
  standard_split_v3/tempsal_erp_seed20260712/checkpoints/best.pt
  standard_split_v3/panoramic_cnn_seed20260710/checkpoints/best.pt
  standard_split_v3/panoramic_cnn_seed20260711/checkpoints/best.pt
  standard_split_v3/panoramic_cnn_seed20260712/checkpoints/best.pt
  standard_split_v3/pathformer3d_train_seed20260710/checkpoints/epoch_0035.pt
  standard_split_v3/pathformer3d_train_seed20260711/checkpoints/epoch_0040.pt
  standard_split_v3/pathformer3d_train_seed20260712/checkpoints/epoch_0040.pt
  standard_split_v3/scandmm_train_seed20260710/checkpoints/epoch_0399.pkl
  standard_split_v3/scandmm_train_seed20260711/checkpoints/epoch_0424.pkl
  standard_split_v3/scandmm_train_seed20260712/checkpoints/epoch_0324.pkl
  standard_split_v3/hat_train_seed20260710/checkpoints/step_003000.pt
  standard_split_v3/hat_train_seed20260711/checkpoints/step_003000.pt
  standard_split_v3/hat_train_seed20260712/checkpoints/step_005000.pt
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260710/checkpoints/best.pt
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260711/checkpoints/best.pt
  sitzmann_standard_v2/v12c_hybrid_dwell_rate_seed20260712/checkpoints/best.pt
  sitzmann_standard_v2/semantic_anchor_seed20260711/checkpoints/best.pt
  sitzmann_standard_v2/static_anchor_seed20260711/checkpoints/best.pt
)
for relative in "${CHECKPOINTS[@]}"; do
  copy_file "${ROOT}/runs/${relative}" "${STAGE}/runs/${relative}"
done

# Remove LaTeX/editor build debris while retaining source figures and tables.
find "${STAGE}" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.git' \) -prune -exec rm -rf {} +
find "${STAGE}/paper" -type f \( \
  -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.fdb_latexmk' \
  -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o -name '*.synctex.gz' \
\) -delete

cat_notice="${STAGE}/PACKAGE_SCOPE.txt"
printf '%s\n' \
  'This package contains code, documentation, paper artifacts, result reports,' \
  'public baseline sources, and validation-selected checkpoints.' \
  'Bulk datasets and prediction caches are intentionally excluded.' \
  'Read docs/PAPER_REPRODUCTION_PACKAGE.md before reproducing experiments.' \
  > "${cat_notice}"

# Per-file checksums are relative to the package root.
(
  cd "${STAGE}"
  find . -type f ! -name SHA256SUMS ! -name FILE_MANIFEST.tsv -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  {
    printf 'bytes\tsha256\tpath\n'
    while read -r digest path; do
      path="${path#\*}"
      path="${path#  }"
      printf '%s\t%s\t%s\n' "$(stat -c '%s' "${path}")" "${digest}" "${path}"
    done < SHA256SUMS
  } > FILE_MANIFEST.tsv
)

rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar --zstd -cf "${ARCHIVE}" -C "${OUT_ROOT}" "${NAME}"
(
  cd "${OUT_ROOT}"
  sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256"
)

echo "Package directory: ${STAGE}"
echo "Archive: ${ARCHIVE}"
du -sh "${STAGE}" "${ARCHIVE}"
