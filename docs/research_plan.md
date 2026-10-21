# Research roadmap (new project)

## Why a new repository?

Keep previous synthetic study (its datasets, canonical tables and plots) immutable. This repository isolates new benchmark implementations, long-context prompts, cross-language and cross-family design, and later method development.

## Stage 0 — replication of the previous code path

- Use working Qwen3-1.7B weights on a V100, FP16 weights, batch=1.
- Reuse same HQQ parameters and direct `model(..., past_key_values=cache)` code path.
- Fix dataset sample IDs, tokenized prompts and tokenizer hash before running treatments.
- Confirm Dynamic smoke works, including sensible nonempty textual answers; inspect truncation artifacts.

**Gate:** no scientific evaluation until Dynamic succeeds end-to-end on a small sample.

## Stage 1 — LongBench v1 small controlled pilot

- `passage_retrieval_en`: closest to synthetic recall; official retrieval_score.
- `multifieldqa_en`: English natural document QA; official qa_f1_score.
- `multifieldqa_zh`: Chinese document QA; official qa_f1_zh_score.
- 40 randomly sampled instances per dataset, fixed seed 2026; Dynamic vs INT4 g16/g64.
- Cap at 16K input tokens; preserve original lengths, truncation flag, prompt hashes.
- Pair scores by identical task ID and token hash.

**Gate:** determine whether the quality gap is visible for any dataset, and whether Dynamic is comfortably above floor.

## Stage 2 — increased power + group size

- Increase n (or evaluate whole chosen subsets).
- Add INT4 g32/g128 only where results are informative.
- Report paired example-level deltas + bootstrap uncertainty and context-length bins.
- Re-run any anomalous cells on new examples; do not repeatedly select only convenient tasks.
- Measure full GPU/latency tradeoff on selected long contexts after quality validation.

## Stage 3 — independent model family

- Choose an **architecturally different** model with feasible V100 memory/context/KV API.
- Define matching size range and same LongBench instances.
- Control max prompt length, prompt formatting and quantization settings.
- Compare within-model Dynamic→INT4 degradation. Absolute baseline score is not directly model-family effect.
- DeepSeek-R1-Distill-Qwen is Qwen-based (not an independent architecture); confirm the backbone before labeling family.

## Stage 4 — language

- First compare English and Chinese subsets **descriptively** using within-dataset deltas.
- For a proper language-controlled study create/curate parallel English–Chinese tasks with matched facts and controlled token-length distributions.
- Investigate tokenizer tokenization density as a possible confound.

## Candidate next mechanism work

Locate quantization sensitivity by layers, heads, and K/V tensor; evaluate smaller groups selectively in sensitive layers while retaining g64 for others. Target: quality close to INT4 g16 with memory closer to g64.

## A useful first research update to the supervisor

1. A brief protocol and version/hash manifest.
2. A table of three datasets x three cache settings with exact n and official metric.
3. Paired deltas and examples of failures; no cross-dataset score averages.
4. What differs from earlier synthetic results, and likely alternative explanations.
5. Proposed next hypothesis, with explicit rejection criteria.
