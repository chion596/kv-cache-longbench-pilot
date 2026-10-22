#!/usr/bin/env bash
#SBATCH --job-name=lb-llama-smoke
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

: "${OTHER_MODEL_PATH:?Нужно указать OTHER_MODEL_PATH для Llama 3.2 1B Instruct}"
test -f "${OTHER_MODEL_PATH}/config.json"
test -f data/prepared/families/llama32-1b-instruct/manifest.json
test -f data/prepared/families/llama32-1b-instruct/passage_retrieval_en.jsonl

CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
mkdir -p results/families/llama32-1b-instruct/smoke

echo "Проверка совместимости Llama: cache=${CACHE}, limit=3"
python -u -m src.run_quality \
  --model "${OTHER_MODEL_PATH}" \
  --tasks data/prepared/families/llama32-1b-instruct/passage_retrieval_en.jsonl \
  --cache "${CACHE}" \
  --limit 3 \
  --resume \
  --out "results/families/llama32-1b-instruct/smoke/passage_retrieval_en_${CACHE}.jsonl"
