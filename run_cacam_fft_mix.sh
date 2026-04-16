#!/bin/bash
# 运行CACAM_FFT_mix在CalIt2数据集

datasets=(
    #"CalIt2.csv"
    #"CICIDS.csv"
    #"Creditcard.csv"
    #"MSL.csv"
    #"PSM.csv"
    #"SMAP.csv"
    #"Genesis.csv"
    #"NYC.csv"
    #"synthetic_con0.072.csv"
    "synthetic_con0.0494.csv"
    #"synthetic_glo0.0718.csv"
    "synthetic_glo0.048.csv"
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
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_both_config.json" \
        --data-name-list "${dataset}" \
        --model-name "self_impl.CACAM_FFT_mix" \
        --model-hyper-params '{}' \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "CACAM_FFT_mix"
done       
