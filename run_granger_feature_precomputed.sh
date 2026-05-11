#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec ./run_granger_feature_current_window.sh "$@"
