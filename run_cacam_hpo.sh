#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ROOT_DIR="$(pwd)"
CONFIG_PATH="unfixed_detect_label_multi_config.json"
CONFIG_MODEL_NAME="CACAM.CACAM"
MODEL_VARIANT="feature_temporal_transformer"
BASE_SAVE_PATH="label/feature_temporal_transformer_hpo"
SAVE_ROOT="${ROOT_PATH:-$ROOT_DIR}"
RESULT_DIR="${SAVE_ROOT}/result"

JOB_NAME="${JOB_NAME:-CACAM_hpo_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="result/${BASE_SAVE_PATH}/${JOB_NAME}/logs"
SUMMARY_FILE="result/${BASE_SAVE_PATH}/${JOB_NAME}/hpo_results.csv"
mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_FILE")"

OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-affiliation_f}"
DATASETS=("PSM.csv" "Genesis.csv")

declare -A ANOMALY_RATIO_MAP=(
    ["PSM.csv"]=3.0
    ["Genesis.csv"]=0.5
)
declare -A BATCH_SIZE_MAP=(
    ["PSM.csv"]=128
    ["Genesis.csv"]=32
)

LR="${LR:-0.0005}"
N_HEADS="${N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
SEQ_LEN="${SEQ_LEN:-192}"

# 调参空间：只扫 d_model, feature_layers, temporal_layers
if [[ -n "${D_MODELS:-}" ]]; then
    IFS=',' read -r -a D_MODELS <<< "${D_MODELS}"
else
    D_MODELS=(32 64 128 256)
fi
if [[ -n "${FEATURE_LAYERS:-}" ]]; then
    IFS=',' read -r -a FEATURE_LAYERS <<< "${FEATURE_LAYERS}"
else
    FEATURE_LAYERS=(0 1 2)
fi
if [[ -n "${TEMPORAL_LAYERS:-}" ]]; then
    IFS=',' read -r -a TEMPORAL_LAYERS <<< "${TEMPORAL_LAYERS}"
else
    TEMPORAL_LAYERS=(1 2 3)
fi

echo "dataset,d_model,d_ff,feature_layers,temporal_layers,score,metric,run_id,log_file" > "$SUMMARY_FILE"

run_bench() {
    local dataset="$1"
    local d_model="$2"
    local feature_layers="$3"
    local temporal_layers="$4"
    local d_ff="$((d_model * 2))"
    local run_id="$5"

    local dataset_base="${dataset%.csv}"
    local save_path="${BASE_SAVE_PATH}/${JOB_NAME}/${dataset_base}/d${d_model}_f${feature_layers}_t${temporal_layers}_${run_id}"
    local log_file="${LOG_DIR}/${dataset_base}_d${d_model}_f${feature_layers}_t${temporal_layers}_${run_id}.log"
    local anomaly_ratio="${ANOMALY_RATIO_MAP[$dataset]}"
    local batch_size="${BATCH_SIZE_MAP[$dataset]}"

    python ./scripts/run_benchmark.py \
        --config-path "$CONFIG_PATH" \
        --data-name-list "$dataset" \
        --model-name "$CONFIG_MODEL_NAME" \
        --model-hyper-params "{
            \"experiment_model_name\": \"CACAM\",
            \"lr\": ${LR},
            \"model_variant\": \"${MODEL_VARIANT}\",
            \"batch_size\": ${batch_size},
            \"d_model\": ${d_model},
            \"d_ff\": ${d_ff},
            \"n_heads\": ${N_HEADS},
            \"feature_layers\": ${feature_layers},
            \"temporal_layers\": ${temporal_layers},
            \"dropout\": ${DROPOUT},
            \"num_epochs\": ${NUM_EPOCHS},
            \"patience\": ${PATIENCE},
            \"seq_len\": ${SEQ_LEN},
            \"anomaly_ratio\": ${anomaly_ratio}
        }" \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "$save_path" \
        > "$log_file" 2>&1

    local report_file
    report_file=$(ls -t "result/${save_path}"/test_report.* 2>/dev/null | head -n 1 || true)
    if [[ -z "$report_file" ]]; then
        echo "[WARN] no test_report for ${dataset} d=${d_model} f=${feature_layers} t=${temporal_layers}" >&2
        echo "${dataset},${d_model},${d_ff},${feature_layers},${temporal_layers},,,${run_id},${log_file}" >> "$SUMMARY_FILE"
        echo ""
        return
    fi

    local score
    if ! score=$(python - "$report_file" "$OBJECTIVE_METRIC" <<'PY'
import csv
import sys

path = sys.argv[1]
metric = sys.argv[2]

with open(path, newline="") as f:
    rows = list(csv.reader(f))

for row in rows[1:]:
    if len(row) < 3:
        continue
    if row[1] == metric:
        print(row[2])
        raise SystemExit
if len(rows) > 1 and len(rows[1]) >= 3:
    print(rows[1][2])
PY
    ); then
        echo "[WARN] failed to parse ${OBJECTIVE_METRIC} from ${report_file}, dataset=${dataset}" >&2
        score=""
    fi
    echo "${dataset},${d_model},${d_ff},${feature_layers},${temporal_layers},${score},${OBJECTIVE_METRIC},${run_id},${log_file}" >> "$SUMMARY_FILE"
}

run_id=0
for d_model in "${D_MODELS[@]}"; do
    if (( d_model % N_HEADS != 0 )); then
        echo "[SKIP] d_model=${d_model} not divisible by n_heads=${N_HEADS}"
        continue
    fi
    for feature_layers in "${FEATURE_LAYERS[@]}"; do
        for temporal_layers in "${TEMPORAL_LAYERS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                run_id=$((run_id + 1))
                echo "[${run_id}] data=${dataset}, d_model=${d_model}, d_ff=$((d_model * 2)), feature_layers=${feature_layers}, temporal_layers=${temporal_layers}"
                if ! run_bench "$dataset" "$d_model" "$feature_layers" "$temporal_layers" "$run_id"; then
                    echo "[ERR] run failed: data=${dataset}, d_model=${d_model}, f=${feature_layers}, t=${temporal_layers}" >&2
                fi
            done
        done
    done
done

python - "$SUMMARY_FILE" "$OBJECTIVE_METRIC" "$NUM_EPOCHS" "$PATIENCE" <<'PY'
import csv
import sys
from collections import defaultdict

summary_path, metric_name, num_epochs, patience = sys.argv[1:5]

rows = []
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

rows = [r for r in rows if r.get("score", "").strip()]
for r in rows:
    r["score"] = float(r["score"])

def format_row(r):
    return (
        f"[{r['dataset']}] d_model={r['d_model']}, "
        f"feature_layers={r['feature_layers']}, temporal_layers={r['temporal_layers']}, "
        f"d_ff={r['d_ff']} => {metric_name}={r['score']:.6f}, log={r['log_file']}"
    )

print("\n=== Best by dataset ===")
by_dataset = defaultdict(list)
for r in rows:
    by_dataset[r["dataset"]].append(r)

for dataset in sorted(by_dataset):
    best = sorted(by_dataset[dataset], key=lambda x: x["score"], reverse=True)[0]
    print(format_row(best))

agg = defaultdict(list)
for r in rows:
    key = (r["d_model"], r["d_ff"], r["feature_layers"], r["temporal_layers"])
    agg[key].append(r["score"])

print("\n=== Best overall (average on all datasets) ===")
full = [
    (k, sum(v) / len(v))
    for k, v in agg.items()
    if len(v) >= 3
]
if full:
    k, v = sorted(full, key=lambda x: x[1], reverse=True)[0]
    d_model, d_ff, feature_layers, temporal_layers = k
    print(
        f"d_model={d_model}, d_ff={d_ff}, feature_layers={feature_layers}, "
        f"temporal_layers={temporal_layers}, {metric_name} mean={v:.6f}"
    )
else:
    print("No complete triples for all datasets, please check failed runs.")

print(f"\nSummary saved: {summary_path}")
print(f"Search config: epochs={num_epochs}, patience={patience}, metric={metric_name}")
PY
