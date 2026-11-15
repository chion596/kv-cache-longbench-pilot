#!/usr/bin/env bash
#SBATCH --job-name=lb-mixed-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-3%2
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'mixed_status=$?; printf "Ошибка mixed: job=%s cell=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$mixed_status" >&2; exit "$mixed_status"' ERR
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MIXED_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "$((SLURM_ARRAY_TASK_ID / 2))" in
  0) MIXED_NAME=qwen3-1.7b; MIXED_MODEL="${QWEN17_MODEL_PATH:-${MIXED_ROOT}/Qwen3-1.7B}" ;;
  1) MIXED_NAME=qwen3-4b; MIXED_MODEL="${QWEN4_MODEL_PATH:-${MIXED_ROOT}/Qwen3-4B}" ;;
  *) echo "Неверный индекс модели" >&2; exit 1 ;;
esac
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  MIXED_DATASET=passage_retrieval_en
else
  MIXED_DATASET=hotpotqa
fi
printf 'Mixed precision: job=%s phase=%s host=%s model=%s dataset=%s\n' \
  "${SLURM_JOB_ID}" "${PHASE:-smoke}" "${HOSTNAME:-unknown}" "${MIXED_MODEL}" "${MIXED_DATASET}"
python -u -m src.run_mixed_precision --name "${MIXED_NAME}" --dataset "${MIXED_DATASET}" \
  --model "${MIXED_MODEL}" --phase "${PHASE:-smoke}"
