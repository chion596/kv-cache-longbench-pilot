#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != kvcache ]]; then
  echo "Сначала: module load python; conda activate kvcache" >&2
  exit 1
fi
ZH_PREFLIGHT_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m src.check_zh_confirmation \
  --qwen17 "${QWEN17_MODEL_PATH:-${ZH_PREFLIGHT_ROOT}/Qwen3-1.7B}" \
  --qwen4 "${QWEN4_MODEL_PATH:-${ZH_PREFLIGHT_ROOT}/Qwen3-4B}" \
  --output "${ZH_PREFLIGHT_OUTPUT:-results/zh_confirmation_preflight_2026-11-17.json}"
