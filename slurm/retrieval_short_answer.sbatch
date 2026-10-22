#!/usr/bin/env bash
#SBATCH --job-name=lb-short8
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
: "${MODEL_PATH:?Укажи MODEL_PATH на локальные веса Qwen3-1.7B}"

# Контроль фиксирует исходные 40 задач: меняется только длина ответа.
# Датасет: passage_retrieval_en; максимум генерации: 8 токенов.
CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
RESULT_DIR="results/short_answer/8tokens"
mkdir -p "${RESULT_DIR}"

echo "Контроль длины ответа: dataset=passage_retrieval_en cache=${CACHE} max_new_tokens=8"
python -u -m src.run_quality \
  --model "${MODEL_PATH}" \
  --tasks "data/prepared/passage_retrieval_en.jsonl" \
  --cache "${CACHE}" \
  --max-new-tokens-override 8 \
  --resume \
  --out "${RESULT_DIR}/passage_retrieval_en_${CACHE}.jsonl"
