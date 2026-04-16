#!/bin/bash
# 运行CACAM在ASD系列数据集

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cacam

# ASD系列数据集 (1-12)
asd_datasets=(
    "ASD_dataset_1.csv"
    "ASD_dataset_2.csv"
    "ASD_dataset_3.csv"
    "ASD_dataset_4.csv"
    "ASD_dataset_5.csv"
    "ASD_dataset_6.csv"
    "ASD_dataset_7.csv"
    "ASD_dataset_8.csv"
    "ASD_dataset_9.csv"
    "ASD_dataset_10.csv"
    "ASD_dataset_11.csv"
    "ASD_dataset_12.csv"
)

for dataset in "${asd_datasets[@]}"; do
    echo "Running CACAM on ${dataset}..."
    python ./scripts/run_benchmark.py \
        --config-path "unfixed_detect_both_config.json" \
        --data-name-list "${dataset}" \
        --model-name "self_impl.CACAM" \
        --model-hyper-params '{}' \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "CACAM_FFT"
    echo "Done: ${dataset}"
done

echo "All ASD datasets completed!"
