#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_PATH="${CONFIG_PATH:-unfixed_detect_label_multi_config.json}"
SAVE_PATH="${SAVE_PATH:-label/granger_feature_precomputed}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TIMEOUT="${TIMEOUT:-60000}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
SEQ_LEN="${SEQ_LEN:-192}"
GRANGER_LAG="${GRANGER_LAG:-3}"
GRANGER_ALPHA="${GRANGER_ALPHA:-0.05}"

run_one() {
    local index="$1"
    local total="$2"
    local data_name="$3"
    local anomaly_ratio="$4"
    local batch_size="$5"
    local d_model="$6"
    local d_ff="$7"
    local n_heads="$8"
    local feature_layers="$9"
    local temporal_layers="${10}"
    local dataset_base="${data_name%.csv}"

    echo "[${index}/${total}] granger_feature_temporal_transformer precomputed Granger on ${data_name}"
    python ./scripts/run_benchmark.py \
        --config-path "${CONFIG_PATH}" \
        --data-name-list "${data_name}" \
        --model-name "CACAM.CACAM" \
        --model-hyper-params "{
            \"lr\": 0.0005,
            \"model_variant\": \"granger_feature_temporal_transformer\",
            \"experiment_model_name\": \"CACAM_granger_precomputed_${dataset_base}\",
            \"batch_size\": ${batch_size},
            \"d_model\": ${d_model},
            \"d_ff\": ${d_ff},
            \"n_heads\": ${n_heads},
            \"feature_layers\": ${feature_layers},
            \"temporal_layers\": ${temporal_layers},
            \"precompute_granger\": true,
            \"dynamic_granger\": false,
            \"granger_lag\": ${GRANGER_LAG},
            \"granger_alpha\": ${GRANGER_ALPHA},
            \"granger_standardize\": true,
            \"granger_debug_print\": false,
            \"granger_mask_type\": \"soft\",
            \"granger_bias\": 2.0,
            \"dropout\": 0.1,
            \"num_epochs\": ${NUM_EPOCHS},
            \"patience\": ${PATIENCE},
            \"seq_len\": ${SEQ_LEN},
            \"anomaly_ratio\": ${anomaly_ratio}
        }" \
        --num-workers "${NUM_WORKERS}" \
        --timeout "${TIMEOUT}" \
        --save-path "${SAVE_PATH}/${dataset_base}"
}

run_one 1 2 "PSM.csv" 3.0 128 32 64 8 1 1
run_one 2 2 "Genesis.csv" 0.5 32 256 512 8 1 3
