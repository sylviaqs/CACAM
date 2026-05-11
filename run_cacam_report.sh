#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash ./run_cacam.sh
bash ./run_ablation.sh
bash ./run_layer_sweep.sh
