#!/usr/bin/env bash
#SBATCH --job-name=lb-zh-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-1%2
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'zh_status=$?; printf "Ошибка ZH: job=%s model=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "$LINENO" "$zh_status" >&2; exit "$zh_status"' ERR
module purge
module load python
conda activate kvcache
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ZH_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "$SLURM_ARRAY_TASK_ID" in
  0) ZH_NAME=qwen3-1.7b; ZH_MODEL="${QWEN17_MODEL_PATH:-$ZH_ROOT/Qwen3-1.7B}" ;;
  1) ZH_NAME=qwen3-4b; ZH_MODEL="${QWEN4_MODEL_PATH:-$ZH_ROOT/Qwen3-4B}" ;;
  *) echo "Неверный индекс модели ZH" >&2; exit 1 ;;
esac
printf 'ZH: job=%s phase=%s host=%s model=%s residual=1024\n' \
  "$SLURM_JOB_ID" "${PHASE:-smoke}" "${HOSTNAME:-unknown}" "$ZH_MODEL"
python -u -m src.run_zh_confirmation --name "$ZH_NAME" --model "$ZH_MODEL" --phase "${PHASE:-smoke}"
