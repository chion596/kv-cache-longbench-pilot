#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXPANSION_PHASE="${1:-smoke}"
if [[ "${EXPANSION_PHASE}" != smoke && "${EXPANSION_PHASE}" != full ]]; then
  echo "Использование: bash tools/start_expansion.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if expansion_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="${expansion_commit}"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
export SMOL_MODEL_PATH="${SMOL_MODEL_PATH:-${KV_MODEL_ROOT}/SmolLM3-3B}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EXPANSION_AUDIT="${EXPANSION_AUDIT_PATH:-results/expansion_audit_2026-11-11.json}"
EXPANSION_NAMES=(qwen3-1.7b qwen3-4b smollm3-3b)
EXPANSION_PATHS=("${QWEN17_MODEL_PATH}" "${QWEN4_MODEL_PATH}" "${SMOL_MODEL_PATH}")
python -m unittest discover -s tests
for i in 0 1 2; do
  if [[ ! -f "data/prepared/expansion_v1/${EXPANSION_NAMES[$i]}/manifest.json" ]]; then
    if [[ "${EXPANSION_PHASE}" == full ]]; then
      echo "Нет готовых входов; сначала нужен smoke" >&2
      exit 1
    fi
    python -m src.prepare_expansion --name "${EXPANSION_NAMES[$i]}" --model "${EXPANSION_PATHS[$i]}" --audit "${EXPANSION_AUDIT}"
  fi
  for offset in 0 8; do
    python -m src.run_expansion --model "${EXPANSION_PATHS[$i]}" --cell "$((i * 16 + offset))" \
      --phase "${EXPANSION_PHASE}" --check-data
  done
done
if [[ "${EXPANSION_PHASE}" == full ]]; then
  python -m src.score_expansion --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=${EXPANSION_PHASE}" slurm/expansion.sbatch
