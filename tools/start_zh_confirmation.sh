#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ZH_PHASE="${1:-smoke}"
if [[ "$ZH_PHASE" != smoke && "$ZH_PHASE" != full ]]; then
  echo "Использование: bash tools/start_zh_confirmation.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if zh_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="$zh_commit"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ "$ZH_PHASE" == smoke ]]; then
  python -m src.prepare_zh_confirmation --qwen17 "$QWEN17_MODEL_PATH" --qwen4 "$QWEN4_MODEL_PATH" \
    --preflight "${ZH_PREFLIGHT_JSON:-results/zh_confirmation_preflight_2026-11-17.json}"
elif [[ ! -f data/prepared/zh_confirmation_v1/selection.json ]]; then
  echo "Сначала подготовка и успешный ZH smoke" >&2
  exit 1
fi
python -m src.run_zh_confirmation --name qwen3-1.7b --model "$QWEN17_MODEL_PATH" --phase "$ZH_PHASE" --check-data
python -m src.run_zh_confirmation --name qwen3-4b --model "$QWEN4_MODEL_PATH" --phase "$ZH_PHASE" --check-data
if [[ "$ZH_PHASE" == full ]]; then
  python -m src.score_zh_confirmation --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=$ZH_PHASE" slurm/zh_confirmation.sbatch
