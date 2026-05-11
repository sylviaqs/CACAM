#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_one() {
    local index="$1"
    local total="$2"
    local data_name="$3"
    local anomaly_ratio="$4"
    local batch_size="$5"
    local d_model="$6"
    local d_ff="$7"
    local feature_layers="$8"
    local temporal_layers="$9"

    echo "[${index}/${total}] feature_temporal_transformer on ${data_name}"
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_label_multi_config.json" \
        --data-name-list "${data_name}" \
        --model-name "CACAM.CACAM" \
        --model-hyper-params "{
            \"lr\": 0.0005,
            \"model_variant\": \"granger_feature_temporal_transformer\",
            \"batch_size\": ${batch_size},
            \"d_model\": ${d_model},
            \"d_ff\": ${d_ff},
            \"n_heads\": 8,
            \"feature_layers\": ${feature_layers},
            \"temporal_layers\": ${temporal_layers},
            \"dropout\": 0.1,
            \"num_epochs\": 5,
            \"patience\": 3,
            \"seq_len\": 192,
            \"anomaly_ratio\": ${anomaly_ratio}
        }" \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "label/feature_temporal_transformer"
}

run_one 1 3 "SMAP.csv" 2.0 128 256 512 1 3 # run_id=97 score=0.588497
# run_one 2 3 "PSM.csv" 3.0 128 32 64 1 1 # run_id=11 score=0.593560
# run_one 3 3 "Genesis.csv" 0.5 32 256 512 1 3 # run_id=99 score=0.878566
