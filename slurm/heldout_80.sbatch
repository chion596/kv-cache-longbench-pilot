#!/usr/bin/env bash
#SBATCH --job-name=lb-held80
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-8%3
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_PATH:?Нужно указать MODEL_PATH на Qwen3-1.7B}"
test -f "${MODEL_PATH}/config.json"
test -f "data/prepared/heldout/manifest.json"

DATASETS=(passage_retrieval_en multifieldqa_en multifieldqa_zh)
CACHES=(dynamic int4_g16 int4_g64)
DATASET="${DATASETS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}"
CACHE="${CACHES[$(( SLURM_ARRAY_TASK_ID % 3 ))]}"

echo "Независимая серия held-out: датасет=${DATASET}, cache=${CACHE}"
echo "Модель: ${MODEL_PATH}"
echo "Задание Slurm: ${SLURM_JOB_ID}"
mkdir -p results/heldout/raw

python -u -m src.run_quality \
  --model "${MODEL_PATH}" \
  --tasks "data/prepared/heldout/${DATASET}.jsonl" \
  --cache "${CACHE}" \
  --resume \
  --out "results/heldout/raw/${DATASET}_${CACHE}.jsonl"
