#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# USER CONFIGURATION
# Edit these defaults here, or override them as environment variables.
# Examples:
#   MODE=single INPUT_PATH=/data/case_0001.nii.gz ./run_enhancement.sh
#   MODE=directory INPUT_PATH=/data/ct OUTPUT_DIR=/data/ct_enh ./run_enhancement.sh
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-directory}"  # single | directory | json
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/output}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/runs/latest}"

# Used only when MODE=json.
DATASET_JSON="${DATASET_JSON:-${REPO_ROOT}/configs/dataset_example.json}"
JSON_SPLITS="${JSON_SPLITS:-training,test}"

# Manuscript parameters.
HU_MIN="${HU_MIN:--1000}"
HU_MAX="${HU_MAX:-1000}"
LEVELS="${LEVELS:-2048}"
WAVELET="${WAVELET:-haar}"
BODY_THRESHOLD_HU="${BODY_THRESHOLD_HU:--600}"
ENTROPY_BINS="${ENTROPY_BINS:-256}"
EPS="${EPS:-1e-8}"
SAVE_HU_MIN="${SAVE_HU_MIN:--1024}"
SAVE_HU_MAX="${SAVE_HU_MAX:-3071}"
PRESERVE_OUTSIDE_MASK="${PRESERVE_OUTSIDE_MASK:-true}"

# Run behavior.
OVERWRITE="${OVERWRITE:-true}"
EXPECTED_COUNT="${EXPECTED_COUNT:-}"
PREVIEW_COUNT="${PREVIEW_COUNT:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --report-dir "${REPORT_DIR}"
  --hu-min "${HU_MIN}"
  --hu-max "${HU_MAX}"
  --levels "${LEVELS}"
  --wavelet "${WAVELET}"
  --body-threshold-hu "${BODY_THRESHOLD_HU}"
  --entropy-bins "${ENTROPY_BINS}"
  --eps "${EPS}"
  --save-hu-min "${SAVE_HU_MIN}"
  --save-hu-max "${SAVE_HU_MAX}"
  --preview-count "${PREVIEW_COUNT}"
)

if [[ "${PRESERVE_OUTSIDE_MASK}" == "true" ]]; then
  COMMON_ARGS+=(--preserve-outside-mask)
else
  COMMON_ARGS+=(--no-preserve-outside-mask)
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  COMMON_ARGS+=(--overwrite)
else
  COMMON_ARGS+=(--no-overwrite)
fi

if [[ -n "${EXPECTED_COUNT}" ]]; then
  COMMON_ARGS+=(--expected-count "${EXPECTED_COUNT}")
fi

printf 'Mode: %s\nInput: %s\nOutput: %s\nReports: %s\n' \
  "${MODE}" "${INPUT_PATH}" "${OUTPUT_DIR}" "${REPORT_DIR}"

case "${MODE}" in
  single)
    "${PYTHON_BIN}" -m body_aware_wavelet.cli single \
      --input-file "${INPUT_PATH}" \
      "${COMMON_ARGS[@]}"
    ;;
  directory)
    "${PYTHON_BIN}" -m body_aware_wavelet.cli directory \
      --input-dir "${INPUT_PATH}" \
      "${COMMON_ARGS[@]}"
    ;;
  json)
    "${PYTHON_BIN}" -m body_aware_wavelet.cli json \
      --dataset-json "${DATASET_JSON}" \
      --splits "${JSON_SPLITS}" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    printf 'ERROR: MODE must be single, directory, or json; received "%s".\n' \
      "${MODE}" >&2
    exit 2
    ;;
esac
