#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "${PROJECT_ROOT}/logs"

CONFIG="${1:-${PROJECT_ROOT}/configs/heloc_cgrr.yaml}"
MARKER_DIR="${MARKER_DIR:-${PROJECT_ROOT}/logs/lime_anchor_pipeline_markers}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Configured Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
mkdir -p "$MARKER_DIR"

CURRENT_STEP="initialization"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

on_error() {
  local exit_code=$?
  echo "[$(timestamp)] FAILED: ${CURRENT_STEP} (exit code ${exit_code})"
  exit "$exit_code"
}

run_step() {
  local stage_id="$1"
  local label="$2"
  shift 2
  local marker="${MARKER_DIR}/${stage_id}.done"

  CURRENT_STEP="$label"

  if [[ "$FORCE" != "1" && -f "$marker" ]]; then
    echo "[$(timestamp)] SKIP: ${label} (marker exists: ${marker})"
    return 0
  fi

  if [[ "$FORCE" == "1" && -f "$marker" ]]; then
    echo "[$(timestamp)] FORCE RERUN: ${label} (ignoring marker: ${marker})"
  fi

  echo "[$(timestamp)] START: ${label}"
  "$@"
  {
    echo "completed_at=$(timestamp)"
    echo "stage_id=${stage_id}"
    echo "label=${label}"
    echo "config=${CONFIG}"
    echo "project_root=${PROJECT_ROOT}"
    echo "command=$*"
  } > "$marker"
  echo "[$(timestamp)] DONE: ${label} (marker written: ${marker})"
}

trap on_error ERR

echo "[$(timestamp)] Using config: ${CONFIG}"
echo "[$(timestamp)] Project root: ${PROJECT_ROOT}"
echo "[$(timestamp)] Marker dir: ${MARKER_DIR}"
echo "[$(timestamp)] FORCE: ${FORCE}"
echo "[$(timestamp)] Python: ${PYTHON_BIN}"

CURRENT_STEP="00 Python environment preflight"
"$PYTHON_BIN" scripts/00_check_environment.py
"$PYTHON_BIN" -c 'import xgboost, lime, alibi; print("[env] LIME/Anchors imports OK", flush=True)'

run_step "01_make_splits" "01 make immutable splits and preprocessing metadata" \
  "$PYTHON_BIN" scripts/01_make_splits.py --config "$CONFIG"

run_step "02_train_blackbox" "02 train XGBoost black-box" \
  "$PYTHON_BIN" scripts/02_train_blackbox.py --config "$CONFIG"

run_step "05_make_counterfactuals" "05 make counterfactual evaluation pairs" \
  "$PYTHON_BIN" scripts/05_make_counterfactuals.py --config "$CONFIG"

run_step "09_extract_lime_anchor_explanations_v2" "09 extract label-safe LIME and Anchors explanations (v2)" \
  "$PYTHON_BIN" scripts/09_extract_lime_anchor_explanations.py --config "$CONFIG" --explainer both --mode both

run_step "06_proxy_forward_x_only" "06 proxy forward x-only" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition x-only --mode forward

run_step "06_proxy_counterfactual_x_only" "06 proxy counterfactual x-only" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition x-only --mode counterfactual

run_step "10_proxy_forward_lime_v2" "10 proxy forward LIME with label-safe explanations (v2)" \
  "$PYTHON_BIN" scripts/10_proxy_local_explanations.py --config "$CONFIG" --condition LIME --mode forward

run_step "10_proxy_counterfactual_lime_v2" "10 proxy counterfactual LIME with label-safe explanations (v2)" \
  "$PYTHON_BIN" scripts/10_proxy_local_explanations.py --config "$CONFIG" --condition LIME --mode counterfactual

run_step "10_proxy_forward_anchors_v2" "10 proxy forward Anchors with label-safe explanations (v2)" \
  "$PYTHON_BIN" scripts/10_proxy_local_explanations.py --config "$CONFIG" --condition Anchors --mode forward

run_step "10_proxy_counterfactual_anchors_v2" "10 proxy counterfactual Anchors with label-safe explanations (v2)" \
  "$PYTHON_BIN" scripts/10_proxy_local_explanations.py --config "$CONFIG" --condition Anchors --mode counterfactual

run_step "07_evaluate_local_explanations_v2" "07 evaluate metrics with label-safe local explanations (v2)" \
  "$PYTHON_BIN" scripts/07_evaluate.py --config "$CONFIG"

run_step "08_make_tables_local_explanations_v2" "08 make paper-ready tables with label-safe local explanations (v2)" \
  "$PYTHON_BIN" scripts/08_make_tables.py --config "$CONFIG"

echo "[$(timestamp)] ALL DONE: LIME/Anchors simulatability pipeline finished successfully"
