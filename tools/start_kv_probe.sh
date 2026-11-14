#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROBE_PHASE="${1:-smoke}"
if [[ "${PROBE_PHASE}" != smoke && "${PROBE_PHASE}" != full ]]; then
  echo "Использование: bash tools/start_kv_probe.sh [smoke|full]" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
if probe_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  export PROTOCOL_CODE_COMMIT="${probe_commit}"
fi
export KV_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export QWEN17_MODEL_PATH="${QWEN17_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-1.7B}"
export QWEN4_MODEL_PATH="${QWEN4_MODEL_PATH:-${KV_MODEL_ROOT}/Qwen3-4B}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PROBE_NAMES=(qwen3-1.7b qwen3-4b)
PROBE_MODELS=("${QWEN17_MODEL_PATH}" "${QWEN4_MODEL_PATH}")
for i in 0 1; do
  if [[ ! -f "data/prepared/kv_probe_v1/${PROBE_NAMES[$i]}/manifest.json" ]]; then
    if [[ "${PROBE_PHASE}" == full ]]; then
      echo "Сначала подготовка и успешный smoke" >&2
      exit 1
    fi
    python -m src.prepare_kv_probe --name "${PROBE_NAMES[$i]}" --model "${PROBE_MODELS[$i]}"
  fi
  python -m src.run_kv_probe --name "${PROBE_NAMES[$i]}" --model "${PROBE_MODELS[$i]}" \
    --phase "${PROBE_PHASE}" --check-data
done
if [[ "${PROBE_PHASE}" == full ]]; then
  python -m src.score_kv_probe --phase smoke --check-only
fi
mkdir -p logs
sbatch --export="ALL,PHASE=${PROBE_PHASE}" slurm/kv_probe.sbatch
