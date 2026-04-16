#!/bin/bash
# 运行 CACAM-FFT (带FFT频域分支) 在 CalIt2 数据集

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cacam

echo "Running CACAM-FFT on CalIt2..."

python ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_both_config.json" \
    --data-name-list "CalIt2.csv" \
    --model-name "self_impl.CACAM-FFT" \
    --model-hyper-params '{"seq_len": 100, "d_model": 128, "n_heads": 4, "dropout": 0.1, "causal_method": "pcmci", "causal_max_lag": 3, "causal_pc_alpha": 0.05, "train_epochs": 10, "batch_size": 128, "learning_rate": 1e-4, "lradj": "type1", "patience": 3, "pct_start": 0.3}' \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "CACAM-FFT"

echo "Done: CalIt2"