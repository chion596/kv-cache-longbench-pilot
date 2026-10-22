#!/usr/bin/env bash
#SBATCH --job-name=lb-g32-g128
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-5%3
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_PATH:?Укажи MODEL_PATH на локальные веса Qwen3-1.7B}"
test -f "${MODEL_PATH}/config.json"
test -f data/prepared/heldout/manifest.json

# Шесть независимых GPU-ячеек: три датасета × g32/g128.
# Исходные 80 заданий, формат ответа и параметры HQQ не меняются.
# %3 ограничивает количество одновременных задач этого массива.
DATASETS=(passage_retrieval_en multifieldqa_en multifieldqa_zh)
CACHES=(int4_g32 int4_g128)
DATASET="${DATASETS[$((SLURM_ARRAY_TASK_ID / 2))]}"
CACHE="${CACHES[$((SLURM_ARRAY_TASK_ID % 2))]}"

mkdir -p results/heldout/raw
echo "LongBench: ${DATASET}, ${CACHE}, модель=${MODEL_PATH}"
echo "Slurm: ${SLURM_JOB_ID}"

python -u -m src.run_quality \
  --model "${MODEL_PATH}" \
  --tasks "data/prepared/heldout/${DATASET}.jsonl" \
  --cache "${CACHE}" \
  --resume \
  --out "results/heldout/raw/${DATASET}_${CACHE}.jsonl"
