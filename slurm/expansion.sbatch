#!/usr/bin/env bash
#SBATCH --job-name=lb-expansion-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-47%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'expansion_status=$?; printf "Ошибка expansion: job=%s cell=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$expansion_status" >&2; exit "$expansion_status"' ERR
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EXPANSION_PHASE="${PHASE:-smoke}"
EXPANSION_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "$((SLURM_ARRAY_TASK_ID / 16))" in
  0) EXPANSION_MODEL="${QWEN17_MODEL_PATH:-${EXPANSION_MODEL_ROOT}/Qwen3-1.7B}" ;;
  1) EXPANSION_MODEL="${QWEN4_MODEL_PATH:-${EXPANSION_MODEL_ROOT}/Qwen3-4B}" ;;
  2) EXPANSION_MODEL="${SMOL_MODEL_PATH:-${EXPANSION_MODEL_ROOT}/SmolLM3-3B}" ;;
  *) echo "Неверная ячейка expansion" >&2; exit 1 ;;
esac
printf 'Expansion: job=%s cell=%s phase=%s host=%s model=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${EXPANSION_PHASE}" "${HOSTNAME:-unknown}" "${EXPANSION_MODEL}"
python -u -m src.run_expansion --cell "${SLURM_ARRAY_TASK_ID}" --model "${EXPANSION_MODEL}" --phase "${EXPANSION_PHASE}"
