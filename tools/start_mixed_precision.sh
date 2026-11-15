#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MIXED_PHASE="${1:-smoke}"
if [[ "${MIXED_PHASE}" != smoke && "${MIXED_PHASE}" != full ]]; then
  echo "Использование: bash tools/start_mixed_precision.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if mixed_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="${mixed_commit}"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MIXED_NAMES=(qwen3-1.7b qwen3-4b)
MIXED_MODELS=("${QWEN17_MODEL_PATH}" "${QWEN4_MODEL_PATH}")
for i in 0 1; do
  if [[ ! -f "data/prepared/mixed_precision_v1/${MIXED_NAMES[$i]}/manifest.json" ]]; then
    if [[ "${MIXED_PHASE}" == full ]]; then
      echo "Сначала подготовка и успешный smoke" >&2
      exit 1
    fi
    python -m src.prepare_mixed_precision --name "${MIXED_NAMES[$i]}" --model "${MIXED_MODELS[$i]}"
  fi
  for dataset in passage_retrieval_en hotpotqa; do
    python -m src.run_mixed_precision --name "${MIXED_NAMES[$i]}" --model "${MIXED_MODELS[$i]}" \
      --dataset "${dataset}" --phase "${MIXED_PHASE}" --check-data
  done
done
if [[ "${MIXED_PHASE}" == full ]]; then
  python -m src.score_mixed_precision --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=${MIXED_PHASE}" slurm/mixed_precision.sbatch
