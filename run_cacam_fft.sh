#!/bin/bash
# 运行CACAM在synthetic系列数据集（除了已有结果的con0.0494和glo0.048）
# 使用MPS (Apple Silicon GPU)

datasets=(
    "CalIt2.csv"
    #"CICIDS.csv"
    "Creditcard.csv"
    "MSL.csv"
    "PSM.csv"
    "SMAP.csv"
    "Genesis.csv"
    "NYC.csv"
    #"synthetic_con0.072.csv"
    #"synthetic_glo0.0718.csv"
    #"synthetic_sea0.0482.csv"
    #"synthetic_sea0.0774.csv"
    #"synthetic_sha0.049.csv"
    #"synthetic_sha0.0742.csv"
    #"synthetic_sub_mix0.0574.csv"
    #"synthetic_sub_mix0.089.csv"
    #"synthetic_tre0.0482.csv"
    #"synthetic_tre0.0778.csv" 
)

for dataset in "${datasets[@]}"; do
    echo "Running CACAM on ${dataset}..."
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_both_config.json" \
        --data-name-list "${dataset}" \
        --model-name "self_impl.CACAM" \
        --model-hyper-params '{"causal_method": "pcmci", "causal_max_lag": 3, "causal_pc_alpha": 0.05, "freq_weight": 0.5}' \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "compare/cacam_fft"
    echo "Done: ${dataset}"
done

echo "All synthetic datasets (except con0.0494 and glo0.048) completed!"