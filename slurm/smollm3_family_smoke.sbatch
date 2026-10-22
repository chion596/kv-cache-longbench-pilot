#!/usr/bin/env bash
#SBATCH --job-name=lb-smollm3-smoke
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

: "${OTHER_MODEL_PATH:?Нужен OTHER_MODEL_PATH с локальными весами SmolLM3-3B}"
test -f "${OTHER_MODEL_PATH}/config.json"
test -f "data/prepared/families/smollm3-3b/manifest.json"
test -f "data/prepared/families/smollm3-3b/passage_retrieval_en.jsonl"

# Первый smoke только на трёх одинаковых retrieval-заданиях:
# проверяет, работает ли стандартный HQQ QuantizedCache с SmolLM3.
CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="results/families/smollm3-3b/smoke"
mkdir -p "${OUT_DIR}"
echo "SmolLM3-3B, cache=${CACHE}, model=${OTHER_MODEL_PATH}"

python -u -m src.run_quality \
  --model "${OTHER_MODEL_PATH}" \
  --tasks data/prepared/families/smollm3-3b/passage_retrieval_en.jsonl \
  --cache "${CACHE}" \
  --limit 3 \
  --resume \
  --out "${OUT_DIR}/passage_retrieval_en_${CACHE}.jsonl"
