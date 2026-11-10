#!/usr/bin/env bash
#SBATCH --job-name=lb-residual-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-3%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'residual_status=$?; printf "Ошибка residual: job=%s cell=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$residual_status" >&2; exit "$residual_status"' ERR
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RESIDUAL_PHASE="${PHASE:-smoke}"
RESIDUAL_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "$((SLURM_ARRAY_TASK_ID / 2))" in
  0) RESIDUAL_MODEL="${QWEN17_MODEL_PATH:-${RESIDUAL_MODEL_ROOT}/Qwen3-1.7B}" ;;
  1) RESIDUAL_MODEL="${QWEN4_MODEL_PATH:-${RESIDUAL_MODEL_ROOT}/Qwen3-4B}" ;;
  *) echo "Неверная ячейка residual" >&2; exit 1 ;;
esac
printf 'Residual: job=%s cell=%s phase=%s host=%s model=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${RESIDUAL_PHASE}" "${HOSTNAME:-unknown}" "${RESIDUAL_MODEL}"
python -u -m src.run_residual --cell "${SLURM_ARRAY_TASK_ID}" --model "${RESIDUAL_MODEL}" --phase "${RESIDUAL_PHASE}"
