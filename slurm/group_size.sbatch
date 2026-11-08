#!/usr/bin/env bash
#SBATCH --job-name=lb-g32-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-7%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'g32_status=$?; printf "Ошибка g32: job=%s cell=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$g32_status" >&2; exit "$g32_status"' ERR
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
G32_PHASE="${PHASE:-smoke}"
MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "$((SLURM_ARRAY_TASK_ID / 4))" in
  0) G32_MODEL="${QWEN17_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B}" ;;
  1) G32_MODEL="${QWEN4_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}" ;;
  *) echo "Неверная ячейка g32" >&2; exit 1 ;;
esac
printf 'g32: job=%s cell=%s phase=%s host=%s model=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${G32_PHASE}" "${HOSTNAME:-unknown}" "${G32_MODEL}"
python -u -m src.run_group_size --cell "${SLURM_ARRAY_TASK_ID}" --model "${G32_MODEL}" --phase "${G32_PHASE}"
