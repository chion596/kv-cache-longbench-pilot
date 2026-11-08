#!/usr/bin/env bash
# На login-узле после module load python / conda activate kvcache.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G32_PHASE="${1:-smoke}"
if [[ "${G32_PHASE}" != smoke && "${G32_PHASE}" != full ]]; then
  echo "Использование: bash tools/start_group_size.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if g32_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="${g32_commit}"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
python -m unittest discover -s tests
for offset in 0 2; do
  python -m src.run_group_size --model "${QWEN17_MODEL_PATH}" --cell "${offset}" --phase "${G32_PHASE}" --check-data
  python -m src.run_group_size --model "${QWEN4_MODEL_PATH}" --cell "$((4 + offset))" --phase "${G32_PHASE}" --check-data
done
if [[ "${G32_PHASE}" == full ]]; then
  python -m src.score_group_size --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=${G32_PHASE}" slurm/group_size.sbatch
