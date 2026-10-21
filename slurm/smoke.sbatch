#!/usr/bin/env bash
#SBATCH --job-name=lb-smoke
#SBATCH --partition=rocky
#SBATCH --account=proj_1782
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_a
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module purge
module load python
conda activate kvcache
cd "${SLURM_SUBMIT_DIR}"
: "${MODEL_PATH:?Export MODEL_PATH pointing to local Qwen3-1.7B first}"
python -u -m src.run_quality \
  --model "${MODEL_PATH}" \
  --tasks data/prepared/passage_retrieval_en.jsonl \
  --cache dynamic \
  --limit 3 \
  --out "results/smoke/passage_retrieval_en_dynamic_${SLURM_JOB_ID}.jsonl"
