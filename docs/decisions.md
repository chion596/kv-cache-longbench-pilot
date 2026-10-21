# Methodology decisions, to revisit only in versioned experimental batches

- **Model**: Qwen3-1.7B, because earlier synthetic results showed non-ceiling/non-floor INT4 g64 exact retrieval (14/45) and model fits on one V100.
- **Datasets**: official LongBench v1 archive. Do not train on LongBench test examples.
- **Prompt**: official LongBench config files; no chat template in the first run to preserve continuity with earlier raw-prompt benchmark. Follow-up chat-template comparison is separately labeled.
- **Metrics**: loaded from official `metrics.py`. No home-grown F1/retrieval replacements.
- **Sample**: deterministic uniform sample of size 40, seed 2026. Selected sample IDs are shared across all configurations (fixed once).
- **Input cap**: 16,384 tokens (mid-truncate); report frequency and original token lengths. Cap permits a feasible V100 pilot, but changes the task relative to full-length LongBench on any truncated example.
- **Generation**: greedy, official max generated tokens per dataset, EOS stop. No temperature, sampling, beam search or hidden template changes.
- **HQQ**: `backend=hqq`, `axis_key=axis_value=1`, `residual_length=128`, bitwidth=4, groups g16/g64. Dynamic baseline unquantized KV.
- **Hardware**: one V100 per job, one model per job, batch size 1. 14B latency is deliberately excluded at this stage.
- **Timing**: CUDA synchronized prefill and decode; incidental measurements during quality trials are diagnostics, not an optimized serving benchmark.
- **Statistical summary**: per-dataset paired changes; percentile bootstrap over sampled instances is exploratory only. No causal claim about *language* from different EN/ZH questions.

**Changes after seeing pilot results:** document new experimental batch/configuration before running it. Do not overwrite the original 40-task outputs when changing sample size, prompt policy, max length or model.
