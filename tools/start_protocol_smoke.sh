#!/usr/bin/env bash
# Выполнять на login-узле после module load python / conda activate kvcache.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "kvcache" ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
export SMOL_MODEL_PATH="${SMOL_MODEL_PATH:-${KV_MODEL_ROOT}/SmolLM3-3B}"
NAMES=(qwen3-1.7b qwen3-4b smollm3-3b)
PATHS=("${QWEN17_MODEL_PATH}" "${QWEN4_MODEL_PATH}" "${SMOL_MODEL_PATH}")
python -m unittest discover -s tests
for i in 0 1 2; do
  if [[ ! -f "${PATHS[$i]}/config.json" ]]; then
    echo "Нет локальной модели: ${PATHS[$i]}" >&2
    exit 1
  fi
  if [[ ! -d "data/prepared/protocol_v1/${NAMES[$i]}" ]]; then
    python -m src.prepare_protocol --name "${NAMES[$i]}" --model "${PATHS[$i]}"
  fi
  for offset in 0 6; do
    python -m src.run_protocol --model "${PATHS[$i]}" --cell "$((i * 12 + offset))" \
      --phase smoke --check-data
  done
done
mkdir -p logs
sbatch --export=ALL,PHASE=smoke slurm/protocol.sbatch
