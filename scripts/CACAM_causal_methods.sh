#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="${1:-CalIt2.csv}"
EPOCHS="${2:-3}"
METHODS="${METHODS:-correlation partial_correlation granger identity uniform}"

for method in ${METHODS}; do
  echo "Running CACAM with causal_method=${method} on ${DATA_NAME}"
  python ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_both_config.json" \
    --data-name-list "${DATA_NAME}" \
    --model-name "self_impl.CACAM" \
    --model-hyper-params "{\"seq_len\": 100, \"d_model\": 128, \"n_heads\": 4, \"dropout\": 0.1, \"train_epochs\": ${EPOCHS}, \"batch_size\": 128, \"learning_rate\": 1e-4, \"lradj\": \"type1\", \"patience\": 3, \"pct_start\": 0.3, \"causal_method\": \"${method}\", \"causal_max_lag\": 3}" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "CACAM_causal/${method}"
done
