#!/bin/sh
set -eu
ROOT=/home/rtx4090/code/python/swy/CACAM
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
DATASETS='CICIDS.csv CalIt2.csv Creditcard.csv GECCO.csv Genesis.csv MSL.csv NYC.csv PSM.csv SMAP.csv SWAT.csv synthetic_con0.072.csv synthetic_glo0.048.csv synthetic_sea0.0482.csv synthetic_sha0.0742.csv synthetic_sub_mix0.0574.csv synthetic_tre0.0482.csv'
for dataset in $DATASETS; do
  case "$dataset" in
    CICIDS.csv) ratio='25.0' ;;
    CalIt2.csv) ratio='3.0' ;;
    Creditcard.csv) ratio='3.0' ;;
    GECCO.csv) ratio='1.0' ;;
    Genesis.csv) ratio='0.5' ;;
    MSL.csv) ratio='3.0' ;;
    NYC.csv) ratio='0.1' ;;
    PSM.csv) ratio='2.0' ;;
    SMAP.csv) ratio='1.0' ;;
    SWAT.csv) ratio='3.0' ;;
    synthetic_con0.072.csv) ratio='5.0' ;;
    synthetic_glo0.048.csv) ratio='0.1' ;;
    synthetic_sea0.0482.csv) ratio='0.1' ;;
    synthetic_sha0.0742.csv) ratio='0.1' ;;
    synthetic_sub_mix0.0574.csv) ratio='0.5' ;;
    synthetic_tre0.0482.csv) ratio='25.0' ;;
    *) echo "missing ratio for $dataset" >&2; exit 1 ;;
  esac
  echo "[corr][direct][binary03][detectboth][ratio=$ratio] $dataset"
  "$PY" ./scripts/run_benchmark.py \
    --config-path unfixed_detect_both_multi_config.json \
    --data-name-list "$dataset" \
    --model-name "CACAM_local_bias.CACAM" \
    --model-hyper-params "{\"seq_len\": 100, \"d_model\": 128, \"n_heads\": 4, \"dropout\": 0.1, \"train_epochs\": 10, \"batch_size\": 128, \"learning_rate\": 1e-4, \"lradj\": \"type1\", \"patience\": 3, \"pct_start\": 0.3, \"causal_method\": \"corr\", \"causal_max_lag\": 3, \"anomaly_ratio\": [$ratio], \"causal_bias_mode\": \"direct\", \"causal_binary_threshold\": 0.3}" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "label/cacam_local_bias_ablation/corr/direct_binary03_best_ratio_detectboth"
done
