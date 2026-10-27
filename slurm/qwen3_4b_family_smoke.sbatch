#!/usr/bin/env bash
#SBATCH --job-name=lb-qwen4-smoke
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-2%1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

: "${QWEN4_MODEL_PATH:?Нужен путь к локальным весам Qwen3-4B}"
test -f "${QWEN4_MODEL_PATH}/config.json"
test -f data/prepared/families/qwen3-4b/manifest.json
test -f data/prepared/families/qwen3-4b/passage_retrieval_en.jsonl

# Отдельный контроль размера модели внутри Qwen3:
# 3 cache × 3 smoke-задания.
CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR=results/families/qwen3-4b/smoke
mkdir -p "${OUT_DIR}"

python -u -m src.run_quality \
  --model "${QWEN4_MODEL_PATH}" \
  --tasks data/prepared/families/qwen3-4b/passage_retrieval_en.jsonl \
  --cache "${CACHE}" \
  --limit 3 \
  --resume \
  --out "${OUT_DIR}/passage_retrieval_en_${CACHE}.jsonl"
