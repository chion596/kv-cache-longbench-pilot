#!/usr/bin/env bash
#SBATCH --job-name=lb-kv-probe-v1
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-1%2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
trap 'probe_status=$?; printf "Ошибка KV probe: job=%s model=%s line=%s exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${LINENO}" "$probe_status" >&2; exit "$probe_status"' ERR
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PROBE_MODEL_ROOT="${KV_MODEL_ROOT:-$(ws_find kv-cache)/models}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) PROBE_NAME=qwen3-1.7b; PROBE_MODEL="${QWEN17_MODEL_PATH:-${PROBE_MODEL_ROOT}/Qwen3-1.7B}" ;;
  1) PROBE_NAME=qwen3-4b; PROBE_MODEL="${QWEN4_MODEL_PATH:-${PROBE_MODEL_ROOT}/Qwen3-4B}" ;;
  *) echo "Неверный индекс модели" >&2; exit 1 ;;
esac
printf 'KV probe: job=%s phase=%s host=%s model=%s\n' \
  "${SLURM_JOB_ID}" "${PHASE:-smoke}" "${HOSTNAME:-unknown}" "${PROBE_MODEL}"
python -u -m src.run_kv_probe --name "${PROBE_NAME}" --model "${PROBE_MODEL}" --phase "${PHASE:-smoke}"
