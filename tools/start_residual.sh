#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RESIDUAL_PHASE="${1:-smoke}"
if [[ "${RESIDUAL_PHASE}" != smoke && "${RESIDUAL_PHASE}" != full ]]; then
  echo "Использование: bash tools/start_residual.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if residual_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="${residual_commit}"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
python -m unittest discover -s tests
python -m src.run_residual --model "${QWEN17_MODEL_PATH}" --cell 0 --phase "${RESIDUAL_PHASE}" --check-data
python -m src.run_residual --model "${QWEN4_MODEL_PATH}" --cell 2 --phase "${RESIDUAL_PHASE}" --check-data
if [[ "${RESIDUAL_PHASE}" == full ]]; then
  python -m src.score_residual --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=${RESIDUAL_PHASE}" slurm/residual.sbatch
