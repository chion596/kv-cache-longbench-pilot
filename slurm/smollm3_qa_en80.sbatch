#!/usr/bin/env bash
#SBATCH --job-name=lb-smollm3-qa80
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-2%2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"

: "${OTHER_MODEL_PATH:?Нужен путь к SmolLM3-3B}"
test -f "${OTHER_MODEL_PATH}/config.json"
test -f data/prepared/families/smollm3-3b/multifieldqa_en.jsonl

CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR=results/families/smollm3-3b/qa_en80
mkdir -p "${OUT_DIR}"

# Полный LongBench QA: 80 заданий, без --limit и без изменения
# лимита генерации. Smoke находится в отдельной директории.
python -u -m src.run_quality \
  --model "${OTHER_MODEL_PATH}" \
  --tasks data/prepared/families/smollm3-3b/multifieldqa_en.jsonl \
  --cache "${CACHE}" \
  --resume \
  --out "${OUT_DIR}/multifieldqa_en_${CACHE}.jsonl"
