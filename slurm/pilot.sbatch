#!/usr/bin/env bash
#SBATCH --job-name=lb-pilot
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --array=0-8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
: "${MODEL_PATH:?Export MODEL_PATH pointing to local Qwen3-1.7B first}"

DATASETS=(passage_retrieval_en multifieldqa_en multifieldqa_zh)
CACHES=(dynamic int4_g16 int4_g64)
DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
CACHE=${CACHES[$(( SLURM_ARRAY_TASK_ID % 3 ))]}

echo "Dataset=$DATASET Cache=$CACHE Model=$MODEL_PATH"
python -u -m src.run_quality \
  --model "${MODEL_PATH}" \
  --tasks "data/prepared/${DATASET}.jsonl" \
  --cache "$CACHE" \
  --resume \
  --out "results/raw/${DATASET}_${CACHE}.jsonl"
