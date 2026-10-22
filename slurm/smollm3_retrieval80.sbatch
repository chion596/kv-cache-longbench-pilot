#!/usr/bin/env bash
#SBATCH --job-name=lb-smollm3-r80
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0,1,3%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

: "${OTHER_MODEL_PATH:?Нужно экспортировать OTHER_MODEL_PATH для SmolLM3-3B}"
test -f "${OTHER_MODEL_PATH}/config.json"
test -f "data/prepared/families/smollm3-3b/manifest.json"
test -f "data/prepared/families/smollm3-3b/passage_retrieval_en.jsonl"

# Индексы 0/1/3: Dynamic, g16 и g64 (первая серия).
# Индексы 2/4: g32 и g128 (при необходимости отдельным sbatch).
CACHES=(dynamic int4_g16 int4_g32 int4_g64 int4_g128)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"

OUT_DIR="results/families/smollm3-3b/retrieval80"
mkdir -p "${OUT_DIR}"
echo "SmolLM3-3B · LongBench passage_retrieval_en · n=80 · cache=${CACHE}"
echo "Model path: ${OTHER_MODEL_PATH}"

python -u -m src.run_quality \
  --model "${OTHER_MODEL_PATH}" \
  --tasks "data/prepared/families/smollm3-3b/passage_retrieval_en.jsonl" \
  --cache "${CACHE}" \
  --resume \
  --out "${OUT_DIR}/passage_retrieval_en_${CACHE}.jsonl"
