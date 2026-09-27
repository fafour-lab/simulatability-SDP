#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "${PROJECT_ROOT}/logs"

CONFIG="${PROJECT_ROOT}/configs/heloc_cgrr.yaml"
WITH_PROXY_REFINEMENT="${WITH_PROXY_REFINEMENT:-0}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-proxy-refinement)
      WITH_PROXY_REFINEMENT=1
      ;;
    --without-proxy-refinement)
      WITH_PROXY_REFINEMENT=0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: bash scripts/run_pipeline.sh [config.yaml] [--with-proxy-refinement]" >&2
      exit 2
      ;;
  esac
  shift
done
MARKER_DIR="${MARKER_DIR:-${PROJECT_ROOT}/logs/pipeline_markers}"
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
echo "[$(timestamp)] Proxy-guided refinement: ${WITH_PROXY_REFINEMENT}"

CURRENT_STEP="00 Python environment preflight"
"$PYTHON_BIN" scripts/00_check_environment.py

run_step "01_make_splits" "01 make immutable splits and preprocessing metadata" \
  "$PYTHON_BIN" scripts/01_make_splits.py --config "$CONFIG"

run_step "02_train_blackbox" "02 train XGBoost black-box" \
  "$PYTHON_BIN" scripts/02_train_blackbox.py --config "$CONFIG"

run_step "03_build_surrogate" "03 build decision-tree surrogate and R0 rules" \
  "$PYTHON_BIN" scripts/03_build_surrogate.py --config "$CONFIG"

run_step "03b_make_rule_baselines" "03b make rule baselines" \
  "$PYTHON_BIN" scripts/03b_make_rule_baselines.py --config "$CONFIG"

if [[ "$WITH_PROXY_REFINEMENT" == "1" ]]; then
  run_step "03c_proxy_forward_r0_refine" "03c proxy forward R0 on refinement split" \
    "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition R0 --mode forward --split refine
fi

run_step "04_refine_faithful" "04 refine rules with Qwen Faithful-Refined" \
  "$PYTHON_BIN" scripts/04_refine_rules.py --config "$CONFIG" --source faithfulness --output-name Faithful-Refined

if [[ "$WITH_PROXY_REFINEMENT" == "1" ]]; then
  run_step "04_refine_proxy" "04 refine rules from proxy/black-box disagreements" \
    "$PYTHON_BIN" scripts/04_refine_rules.py --config "$CONFIG" --source proxy --output-name Proxy-Refined
fi

run_step "05_make_counterfactuals" "05 make counterfactual evaluation pairs" \
  "$PYTHON_BIN" scripts/05_make_counterfactuals.py --config "$CONFIG"

run_step "06_proxy_forward_x_only" "06 proxy forward x-only" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition x-only --mode forward

run_step "06_proxy_forward_r0" "06 proxy forward R0" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition R0 --mode forward

run_step "06_proxy_forward_faithful" "06 proxy forward Faithful-Refined" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Faithful-Refined --mode forward

if [[ "$WITH_PROXY_REFINEMENT" == "1" ]]; then
  run_step "06_proxy_forward_proxy_refined" "06 proxy forward Proxy-Refined" \
    "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Proxy-Refined --mode forward
fi

run_step "06_proxy_forward_corrupted_r0" "06 proxy forward Corrupted-R0" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Corrupted-R0 --mode forward

run_step "06_proxy_counterfactual_x_only" "06 proxy counterfactual x-only" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition x-only --mode counterfactual

run_step "06_proxy_counterfactual_r0" "06 proxy counterfactual R0" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition R0 --mode counterfactual

run_step "06_proxy_counterfactual_faithful" "06 proxy counterfactual Faithful-Refined" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Faithful-Refined --mode counterfactual

if [[ "$WITH_PROXY_REFINEMENT" == "1" ]]; then
  run_step "06_proxy_counterfactual_proxy_refined" "06 proxy counterfactual Proxy-Refined" \
    "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Proxy-Refined --mode counterfactual
fi

run_step "06_proxy_counterfactual_corrupted_r0" "06 proxy counterfactual Corrupted-R0" \
  "$PYTHON_BIN" scripts/06_proxy_simulation.py --config "$CONFIG" --condition Corrupted-R0 --mode counterfactual

if [[ "$WITH_PROXY_REFINEMENT" == "1" ]]; then
  run_step "07_evaluate_with_proxy_refinement" "07 evaluate metrics with Proxy-Refined" \
    "$PYTHON_BIN" scripts/07_evaluate.py --config "$CONFIG"

  run_step "08_make_tables_with_proxy_refinement" "08 make paper-ready tables with Proxy-Refined" \
    "$PYTHON_BIN" scripts/08_make_tables.py --config "$CONFIG"
else
  run_step "07_evaluate" "07 evaluate metrics" \
    "$PYTHON_BIN" scripts/07_evaluate.py --config "$CONFIG"

  run_step "08_make_tables" "08 make paper-ready tables" \
    "$PYTHON_BIN" scripts/08_make_tables.py --config "$CONFIG"
fi

echo "[$(timestamp)] ALL DONE: full CGRR pipeline finished successfully"
