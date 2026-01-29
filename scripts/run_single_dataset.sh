#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_single_dataset.sh <config_path> [gpu_id]
# Runs mine -> select -> memory -> eval on a single dataset with one GPU.
# Common config notes:
#   - Default: math500 + DeepSeek-R1-Distill-Qwen-1.5B (CONFIG=configs/default).
#   - To switch dataset/model: copy configs/default, edit task.dataset and model.name_or_path, point CONFIG to the new file.

CONFIG=${1:-configs/amc.yaml}
GPU=${2:-0}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
PY_CMD=${PY_CMD:-python}

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

echo "[run_single_dataset] config=${CONFIG} gpu=${GPU} run_id=${RUN_ID}"

"${PY_CMD}" run.py --config "${CONFIG}" --run-id "${RUN_ID}" mine
"${PY_CMD}" run.py --config "${CONFIG}" --run-id "${RUN_ID}" select
"${PY_CMD}" run.py --config "${CONFIG}" --run-id "${RUN_ID}" memory
"${PY_CMD}" run.py --config "${CONFIG}" --run-id "${RUN_ID}" eval
