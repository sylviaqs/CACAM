#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PSM_GRANGER_G="${PSM_GRANGER_G:-}"
GENESIS_GRANGER_G="${GENESIS_GRANGER_G:-}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

while pgrep -af 'run_ablation.sh|label/ablation/' >/dev/null 2>&1; do
    echo "[WAIT] run_ablation.sh still running, sleep ${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
done

run_one() {
    local index="$1"
    local total="$2"
    local sweep_axis="$3"
    local sweep_value="$4"
    local data_name="$5"
    local anomaly_ratio="$6"
    local batch_size="$7"
    local d_model="$8"
    local d_ff="$9"
    local feature_layers="${10}"
    local temporal_layers="${11}"
    local granger_graph_path="${12:-}"
    local dataset_base="${data_name%.csv}"
    local granger_graph_json=""

    if [[ -n "${granger_graph_path}" ]]; then
        granger_graph_json=", \"granger_graph_path\": \"${granger_graph_path}\""
    fi

    echo "[${index}/${total}] ${dataset_base} ${sweep_axis}=${sweep_value}"
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_label_multi_config.json" \
        --data-name-list "${data_name}" \
        --model-name "CACAM.CACAM" \
        --model-hyper-params "{
            \"lr\": 0.0005,
            \"model_variant\": \"granger_feature_temporal_transformer\",
            \"experiment_model_name\": \"CACAM_sweep_${dataset_base}_${sweep_axis}_${sweep_value}\",
            \"batch_size\": ${batch_size},
            \"d_model\": ${d_model},
            \"d_ff\": ${d_ff},
            \"n_heads\": 8,
            \"feature_layers\": ${feature_layers},
            \"temporal_layers\": ${temporal_layers},
            \"granger_mask_type\": \"soft\",
            \"granger_bias\": 2.0,
            \"dropout\": 0.1,
            \"num_epochs\": 5,
            \"patience\": 3,
            \"seq_len\": 192,
            \"anomaly_ratio\": ${anomaly_ratio}${granger_graph_json}
        }" \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "label/layer_sweep/${dataset_base}/${sweep_axis}"
}

run_dataset() {
    local data_name="$1"
    local anomaly_ratio="$2"
    local batch_size="$3"
    local d_model="$4"
    local d_ff="$5"
    local base_temporal_layers="$6"
    local granger_graph_path="$7"
    local total=8
    local idx=1

    for feature_layers in 0 1 2 3; do
        run_one "$idx" "$total" "feature_layers" "$feature_layers" "$data_name" "$anomaly_ratio" "$batch_size" "$d_model" "$d_ff" "$feature_layers" "$base_temporal_layers" "$granger_graph_path"
        idx=$((idx + 1))
    done

    for temporal_layers in 1 2 3 4; do
        run_one "$idx" "$total" "temporal_layers" "$temporal_layers" "$data_name" "$anomaly_ratio" "$batch_size" "$d_model" "$d_ff" 1 "$temporal_layers" "$granger_graph_path"
        idx=$((idx + 1))
    done
}

run_dataset "PSM.csv" 3.0 128 32 64 1 "${PSM_GRANGER_G}"
run_dataset "Genesis.csv" 0.5 32 256 512 3 "${GENESIS_GRANGER_G}"

python - <<'PY'
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

records = defaultdict(dict)
for report_path in glob.glob('result/label/layer_sweep/**/test_report.*.csv', recursive=True):
    with open(report_path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 3 or ';' not in rows[0][2]:
        continue
    model_name, payload = rows[0][2].split(';', 1)
    if model_name != 'ours':
        continue
    cfg = json.loads(payload)
    exp = cfg.get('experiment_model_name', '')
    if not exp.startswith('CACAM_sweep_'):
        continue
    metrics = {row[1]: float(row[2]) for row in rows[1:] if len(row) >= 3 and row[1] in {'affiliation_f', 'auc_roc'} and row[2] not in {'', None}}
    if len(metrics) != 2:
        continue
    stat = os.stat(report_path)
    old = records[exp].get('_mtime')
    if old is None or stat.st_mtime > old:
        records[exp] = {'metrics': metrics, '_mtime': stat.st_mtime}

summary_rows = []
series = defaultdict(lambda: defaultdict(list))
for exp, payload in records.items():
    body = exp[len('CACAM_sweep_'):]
    dataset, axis, value = body.rsplit('_', 2)
    value = int(value)
    metrics = payload['metrics']
    summary_rows.append({
        'dataset': dataset,
        'axis': axis,
        'value': value,
        'affiliation_f': metrics['affiliation_f'],
        'auc_roc': metrics['auc_roc'],
    })
    series[dataset][axis].append((value, metrics['affiliation_f'], metrics['auc_roc']))

summary_rows.sort(key=lambda x: (x['dataset'], x['axis'], x['value']))
os.makedirs('result/label/layer_sweep/charts', exist_ok=True)
with open('result/label/layer_sweep/charts/layer_sweep_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'axis', 'value', 'affiliation_f', 'auc_roc'])
    writer.writeheader()
    writer.writerows(summary_rows)

for dataset, axes in series.items():
    for axis, rows in axes.items():
        rows = sorted(rows)
        x = [item[0] for item in rows]
        aff = [item[1] for item in rows]
        auc = [item[2] for item in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(x, aff, marker='o', label='affiliation_f')
        plt.plot(x, auc, marker='s', label='auc_roc')
        plt.xlabel(axis)
        plt.ylabel('score')
        plt.title(f'{dataset} {axis} sweep')
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/label/layer_sweep/charts/{dataset}_{axis}.png', dpi=150)
        plt.close()
PY
