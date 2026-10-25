#!/usr/bin/env bash
#SBATCH --job-name=lb-smollm3-qa-smoke
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

: "${OTHER_MODEL_PATH:?Нужны локальные веса SmolLM3-3B}"
test -f "${OTHER_MODEL_PATH}/config.json"
test -f data/prepared/families/smollm3-3b/multifieldqa_en.jsonl

CACHES=(dynamic int4_g16 int4_g64)
CACHE="${CACHES[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR=results/families/smollm3-3b/qa_en_smoke
mkdir -p "${OUT_DIR}"

# Восемь одинаковых заданий: сначала проверяем качество raw QA.
python -u -m src.run_quality \
  --model "${OTHER_MODEL_PATH}" \
  --tasks data/prepared/families/smollm3-3b/multifieldqa_en.jsonl \
  --cache "${CACHE}" \
  --limit 8 \
  --resume \
  --out "${OUT_DIR}/multifieldqa_en_${CACHE}.jsonl"
