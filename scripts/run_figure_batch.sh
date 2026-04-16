#!/usr/bin/env bash

set +e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

FAIL_LOG="$ROOT_DIR/result/figure_batch_failures.log"
mkdir -p "$(dirname "$FAIL_LOG")"
: > "$FAIL_LOG"

run_existing_script() {
  local task="$1"
  local dataset_dir="$2"
  local script_name="$3"
  local script_dir="$ROOT_DIR/scripts/multivariate_detection/${task}/${dataset_dir}"
  local script_path="$script_dir/${script_name}.sh"

  if [ ! -f "$script_path" ]; then
    if [ "$script_name" = "IsolationForest" ] && [ -f "$script_dir/isolationforestski.sh" ]; then
      script_path="$script_dir/isolationforestski.sh"
    else
      echo "missing $script_path" | tee -a "$FAIL_LOG"
      return 0
    fi
  fi

  echo "running $task $dataset_dir $script_name"
  if ! sh "$script_path"; then
    echo "failed $task $dataset_dir $script_name" | tee -a "$FAIL_LOG"
  fi
}

DATASET_DIRS=(
  "Genesis_script"
  "CICIDS_script"
  "Creditcard_script"
  "MSL_script"
  "PSM_script"
  "SMAP_script"
  "CalIt2_script"
)

SCRIPT_MODELS=(
  "CATCH"
  "DUET"
  "AnomalyTransformer"
  "AutoEncoder"
  "pcaodetectorski"
  "ocsvmski"
  "IsolationForest"
  "hbosski"
  "CACAM"
)

for dataset_item in "${DATASET_DIRS[@]}"; do
  dataset_dir="$dataset_item"

  for script_name in "${SCRIPT_MODELS[@]}"; do
    run_existing_script "detect_label" "$dataset_dir" "$script_name"
    run_existing_script "detect_score" "$dataset_dir" "$script_name"
  done
done

python ./scripts/extract_figure_table.py

echo "done"
echo "failure log: $FAIL_LOG"
