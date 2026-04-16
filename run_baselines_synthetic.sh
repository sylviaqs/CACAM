#!/bin/bash
# 运行三个baseline模型在synthetic系列数据集
# IF (IsolationForest), HBOS, OCSVM

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cacam

# synthetic系列数据集（去除已有的con0.0494和glo0.048）
synthetic_datasets=(
    "synthetic_con0.072.csv"
    "synthetic_glo0.0718.csv"
    "synthetic_sea0.0482.csv"
    "synthetic_sea0.0774.csv"
    "synthetic_sha0.049.csv"
    "synthetic_sha0.0742.csv"
    "synthetic_sub_mix0.0574.csv"
    "synthetic_sub_mix0.089.csv"
    "synthetic_tre0.0482.csv"
    "synthetic_tre0.0778.csv"
)

# 三个baseline模型
models=(
    "merlion.IsolationForest:label/IsolationForest"
    "tods.hbosski:label/hbosski"
    "tods.ocsvmski:label/ocsvmski"
)

for dataset in "${synthetic_datasets[@]}"; do
    for model_info in "${models[@]}"; do
        model_name="${model_info%%:*}"
        save_path="${model_info##*:}"

        echo "Running ${model_name} on ${dataset}..."
        python ./scripts/run_benchmark.py \
            --config-path "unfixed_detect_label_multi_config.json" \
            --data-name-list "${dataset}" \
            --model-name "${model_name}" \
            --model-hyper-params '{}' \
            --num-workers 1 \
            --timeout 60000 \
            --save-path "${save_path}"
        echo "Done: ${model_name} on ${dataset}"
    done
done

echo "All baseline models on synthetic datasets completed!"
