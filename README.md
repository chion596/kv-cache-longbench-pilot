# KV-cache robustness on LongBench — pilot

New, independent continuation of the previous [Qwen3 KV-cache research](https://github.com/chion596/kv-cache-efficiency).

**Objective:** test whether the strong `q_group_size` dependence observed on synthetic passkey retrieval is also present on standard long-context tasks. The first controlled experiment uses Qwen3-1.7B and three LongBench v1 subsets (`passage_retrieval_en`, `multifieldqa_en`, `multifieldqa_zh`), comparing `DynamicCache`, stock HQQ `INT4 g16` and `INT4 g64`.

This is an **exploratory pilot subset**, **not** a complete LongBench leaderboard submission. Never present pilot percentages as the published full LongBench score.

## Hypothesis

Within the *same model, dataset and task instances*, `INT4 g16` will suffer a smaller quality drop relative to DynamicCache than `INT4 g64` on at least some long-context tasks. Whether this transfers from synthetic passkey retrieval to QA and from English to Chinese remains an **open question**; no LongBench results have been collected yet.

## Design

| Dimension | Pilot |
|---|---|
| Model | Qwen3-1.7B, FP16 weights, greedy decode, batch size 1 |
| GPU | Single HSE NVIDIA V100-SXM2 32GB |
| Baseline | Transformers `DynamicCache` |
| Treatments | Transformers stock `QuantizedCache` + HQQ, INT4, `q_group_size=16/64` |
| Other HQQ parameters | `axis_key=1`, `axis_value=1`, `residual_length=128` |
| Dataset | Original LongBench v1; 3 named subsets |
| Subsample | 40 samples per subset, random seed 2026 (fixed sampled IDs) |
| Prompt | Official LongBench `dataset2prompt.json`; raw prompt, no chat template |
| Max output tokens | Official LongBench `dataset2maxlen.json` per dataset |
| Context cap | 16,384 *input tokens*, retain first and last halves on truncation |
| Evaluation | Imported **official** `LongBench/metrics.py` |
| Comparison | Each cache sees exactly the same token IDs; paired difference vs Dynamic |

**Important caveat:** LongBench's official reference inference code often truncates via decoded text when prompts exceed a model's context; our cap is applied on token IDs. We log original token count and truncation for every task. Therefore this pilot uses the official **datasets, prompt templates and evaluation metrics**, but its generation/truncation protocol is a **controlled custom adaptation**, not a byte-for-byte reproduction of an existing LongBench leaderboard run.

### Expected outputs

- Per-example model outputs with identical `task_id` and `input_ids_sha256` for all cache configurations.
- Official per-dataset scores (`results/summary/dataset_scores.csv`).
- Paired quality changes in percentage points against Dynamic (`results/summary/paired_deltas.csv`).
- A basic plot (`plots/pilot_scores.png`).
- Timing and GPU peak allocation diagnostics for every example (not a dedicated performance benchmark).

## Setup on HSE cluster (login node)

The examples below assume you cloned this repository to `~/kv-cache-longbench-pilot` and already have the `kvcache` conda environment and Qwen3-1.7B weights from the previous project. Run commands **in order, on the cluster login node**, unless otherwise noted.

```bash
cd ~/kv-cache-longbench-pilot
module load python
conda activate kvcache
python -m pip install -r requirements-extra.txt

# Upstream LongBench is a separate dependency; its code/data are NOT vendored.
git clone https://github.com/THUDM/LongBench.git external/LongBench
# Record the version actually evaluated:
git -C external/LongBench rev-parse HEAD

# Reuse EXISTING model weights; do not re-download the 1.7B model.
source cluster.env.example
ls "$MODEL_PATH/config.json"

# Downloads the original official data.zip from Hugging Face (requires network).
python -m src.download_data
```

If outbound network is blocked from the login node, download/clone these upstream resources on your own computer and transfer the `data.zip` and `external/LongBench` directory to the cluster using your existing SCP workflow.

### First: 3-example Dynamic smoke test

```bash
cd ~/kv-cache-longbench-pilot
source cluster.env.example
python -m src.prepare_data --model "$MODEL_PATH" --n 5 --max-input-tokens 16384
python -m unittest discover -s tests -v
mkdir -p logs results/smoke results/raw results/summary
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/smoke.sbatch
```

Check the job via `squeue -u "$USER"` and then inspect `logs/lb-smoke-<JOBID>.out`. The expected result is **three generated predictions and timings**. This verifies data prep, tokenizer, model forward, scoring inputs, and CUDA execution. Do not claim a benchmark result from three examples.

### Then: 40 tasks × 3 datasets × 3 caches

Only after inspecting smoke predictions:

```bash
cd ~/kv-cache-longbench-pilot
source cluster.env.example
python -m src.prepare_data --model "$MODEL_PATH" --n 40 --max-input-tokens 16384
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/pilot.sbatch
```

This submits a Slurm array of **9 independent jobs** (`3 datasets × 3 caches`). Jobs write distinct JSONL files. Each job loads one model copy and processes its selected examples; `--resume` skips already saved IDs but checks their input hashes. Use `squeue -u "$USER"` to watch progress.

### Analysis on login node (no GPU)

Wait until **all 9 jobs** have succeeded. A partial matrix should intentionally trigger a paired-comparison error rather than quietly mix unequal task sets.

```bash
cd ~/kv-cache-longbench-pilot
source cluster.env.example
python -m src.score --longbench-code external/LongBench/LongBench
python -m src.plot
cat results/summary/dataset_scores.csv
cat results/summary/paired_deltas.csv
```

### Verify before any scientific conclusion

Read a few generated predictions for each cache and inspect: official score behavior, early stopping, prompt length, truncation frequency and Qwen3 thinking-marker artifacts. For the first paired trial, make sure DynamicCache performs reasonably; if not, investigate raw/chat formatting and generation before interpreting HQQ changes. Do **not** tune prompt formatting selectively for a cache — the same prompt must be used for all conditions.

## Environment

This scaffold deliberately reuses the *working* GPU environment from the prior project, instead of attempting to reinstall CUDA/PyTorch on the HPC cluster:

- Python 3.11; torch 2.14.0+cu126; Transformers 5.17.0;
- HQQ 0.2.8.post1; FP16 model weights;
- additional upstream metric dependencies in `requirements-extra.txt`.

Record `pip freeze`, `git -C external/LongBench rev-parse HEAD`, `nvidia-smi` and `data/prepared/manifest.json` along with any finalized experiment. Nothing under `data/raw/`, `data/prepared/`, or `results/raw/` is committed by default.

## Repository layout

```
src/download_data.py      download official original LongBench data.zip
src/prepare_data.py       fixed random subset and tokenized prompts
src/run_quality.py        dynamic / HQQ generation; per-example diagnostics
src/score.py              official LongBench metrics; paired bootstrap intervals
src/plot.py               exploratory per-dataset chart
slurm/smoke.sbatch        3 DynamicCache examples on GPU
slurm/pilot.sbatch        9 Slurm jobs for first full pilot
cluster.env.example      environment paths for HSE cluster
docs/research_plan.md     follow-up experimental roadmap
tests/                    offline unit tests
```

## Scientific limitations

1. Pilot of 40 examples per dataset is exploratory; report n and paired uncertainty.
2. Scores are **not** directly comparable across datasets with different metrics.
3. English/Chinese datasets are not parallel translations. Different observed quality losses **do not establish a causal language effect**.
4. Input is capped at 16K and some examples can be middle-truncated; report those counts.
5. Tested quantization overhead is specific to stock HQQ path, V100 hardware and batch size 1.
6. `DynamicCache` itself can perform poorly on some examples; always inspect and use paired changes.
7. Raw prompts without Qwen chat template may not be optimal; any later chat-template variant must be a **separate** controlled experiment.
8. A future cross-family comparison must establish cache API compatibility and architecture details first.

## Upstream references

- [LongBench (THUDM)](https://github.com/THUDM/LongBench) — original dataset, configs, and evaluation code.
- Bai et al., [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://aclanthology.org/2024.acl-long.172/), ACL 2024.
- [LongBench dataset archive](https://huggingface.co/datasets/zai-org/LongBench).
- [Transformers KV cache guide](https://huggingface.co/docs/transformers/v5.17.0/kv_cache).
