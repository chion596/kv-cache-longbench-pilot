#!/usr/bin/env bash
#SBATCH --job-name=lb-protocol-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-35%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'protocol_status=$?; printf "Ошибка Slurm: job=%s array=%s cell=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$protocol_status" >&2; exit "$protocol_status"' ERR
printf 'Slurm: job=%s array=%s cell=%s phase=%s host=%s cwd=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" \
  "${PHASE:-smoke}" "${HOSTNAME:-unknown}" "${SLURM_SUBMIT_DIR}"
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PHASE="${PHASE:-smoke}"
MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
# Порядок соответствует experiments/protocol_v1.json: по 12 ячеек на модель.
case "$((SLURM_ARRAY_TASK_ID / 12))" in
  0) PROTOCOL_MODEL="${QWEN17_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B}" ;;
  1) PROTOCOL_MODEL="${QWEN4_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}" ;;
  2) PROTOCOL_MODEL="${SMOL_MODEL_PATH:-${MODEL_ROOT}/SmolLM3-3B}" ;;
  *) echo "Неверная ячейка" >&2; exit 1 ;;
esac
test -f "${PROTOCOL_MODEL}/config.json"
printf 'Модель: %s\n' "${PROTOCOL_MODEL}"
printf 'Commit при отправке: %s\n' "${PROTOCOL_CODE_COMMIT:-metadata checkout}"
python - <<'PY'
import importlib.metadata
import sys
print("Python:", sys.executable, flush=True)
for package in ("torch", "transformers", "hqq"):
    try:
        print(package, importlib.metadata.version(package), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print(package, "не установлен", flush=True)
PY
python -u -m src.run_protocol --cell "${SLURM_ARRAY_TASK_ID}" \
  --model "${PROTOCOL_MODEL}" --phase "${PHASE}"
