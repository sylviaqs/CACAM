#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PSM_GRANGER_G="${PSM_GRANGER_G:-}"
GENESIS_GRANGER_G="${GENESIS_GRANGER_G:-}"

run_one() {
    local index="$1"
    local total="$2"
    local ablation_name="$3"
    local data_name="$4"
    local anomaly_ratio="$5"
    local batch_size="$6"
    local d_model="$7"
    local d_ff="$8"
    local feature_layers="$9"
    local temporal_layers="${10}"
    local granger_graph_path="${11:-}"
    local granger_graph_json=""

    if [[ -n "${granger_graph_path}" ]]; then
        granger_graph_json=", \"granger_graph_path\": \"${granger_graph_path}\""
    fi

    echo "[${index}/${total}] ${ablation_name} on ${data_name}"
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_label_multi_config.json" \
        --data-name-list "${data_name}" \
        --model-name "CACAM.CACAM" \
        --model-hyper-params "{
            \"lr\": 0.0005,
            \"model_variant\": \"granger_feature_temporal_transformer\",
            \"experiment_model_name\": \"CACAM_abl_${ablation_name}\",
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
        --save-path "label/ablation/${ablation_name}"
}

run_psm() {
    run_one 1 3 "wo_feature" "PSM.csv" 3.0 128 32 64 0 1 ""
    run_one 2 3 "feature_no_granger" "PSM.csv" 3.0 128 32 64 1 1 ""
    run_one 3 3 "feature_granger" "PSM.csv" 3.0 128 32 64 1 1 "${PSM_GRANGER_G}"
}

run_genesis() {
    run_one 1 3 "wo_feature" "Genesis.csv" 0.5 32 256 512 0 3 ""
    run_one 2 3 "feature_no_granger" "Genesis.csv" 0.5 32 256 512 1 3 ""
    run_one 3 3 "feature_granger" "Genesis.csv" 0.5 32 256 512 1 3 "${GENESIS_GRANGER_G}"
}

run_psm
run_genesis
