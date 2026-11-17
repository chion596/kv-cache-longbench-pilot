**mixed_precision_v1: smoke пройден, 15 ноября 2026 года.**

Обновление 17 ноября: [полная серия завершена](mixed_precision_results_2026-11-17.md).
Этот документ сохраняет результаты и решение по проверочному этапу.

Получен полный stdout scorer: **20 ячеек, 60 основных ответов**.
Scorer подтвердил фактическую битность, отсутствие refresh, сверки всех
greedy-траекторий контролей (logits/output IDs) со stock HQQ 4/8 и exact с Dynamic. K8/V4 и K4/V8
имеют равные GPU cache bytes **на каждом входе**, а не только в среднем.
Условия перехода к full выполнены по предоставленному выводу.

[Протокол](mixed_precision_protocol_2026-11-15.md)
· [20 средних CSV](../reports/mixed_precision_smoke_2026-11-15/dataset_scores.csv)
· [Baseline drift](../reports/mixed_precision_smoke_2026-11-15/reference_drift.csv)
· [Происхождение и проверки](../reports/mixed_precision_smoke_2026-11-15/provenance.json).

Предусмотрены 48 дополнительных reference-генераций. Их отдельные traces,
raw JSONL, Slurm logs и runtime provenance не переданы; здесь сверены
агрегаты и сообщения завершившегося scorer, GPU-расчёты локально не повторялись.
Запрошенный runner commit — `7ef10c6`; фактический hash кода кластера
в stdout не указан.

**Память после вопроса, до decode.**

| Модель | Набор | Dynamic, среднее MiB | K8/V4 = K4/V8, среднее MiB | Среднее отношение к Dynamic |
|---|---|---:|---:|---:|
| Qwen3-1.7B | Retrieval EN | 1374.81 | 570.44 | 41.50% |
| Qwen3-1.7B | HotpotQA | 1404.81 | 574.88 | 40.93% |
| Qwen3-4B | Retrieval EN | 1767.61 | 733.43 | 41.50% |
| Qwen3-4B | HotpotQA | 1806.19 | 739.14 | 40.93% |

Включены tensors metadata и точный residual, storages считаются один раз.
Отношение — среднее per-input ratios, а не обязательно отношение двух средних.
Это около 58.5–59.1% меньшего **resident KV cache** на smoke-входах,
не экономия всей GPU RAM. Веса, allocator reserve и временные dequantized
tensors сюда не входят. Номинальный payload 37.5% не равен фактическому
40.9–41.5% с metadata/residual. Равенство budgets относится к 12 парным
model/dataset/input случаям, проверенным scorer; исходные per-input bytes
локально не получены. На full размеры и ratios измеряются снова.

**Качество: только проверочные наблюдения, n=3 на ячейку.**

| Модель | Набор / официальная метрика, % | Dynamic | K4/V4 | K8/V4 | K4/V8 | K16/V4 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Retrieval score | 100 | 0 | 100 | 0 | 100 |
| Qwen3-1.7B | HotpotQA F1 | 33.33 | 0 | 33.33 | 0 | 33.33 |
| Qwen3-4B | Retrieval score | 100 | 33.33 | 100 | 33.33 | 100 |
| Qwen3-4B | HotpotQA F1 | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 |

На retrieval K8/V4 сохраняет средний baseline score, а K4/V8 не устраняет
провал при тех же байтах. Это предварительный сигнал в пользу гипотезы K,
но три случая не устанавливают улучшение на n=80 или общую устойчивость.
На HotpotQA 4B все пять средних одинаковы; smoke не показывает преимущества
K8/V4 по этой метрике. Даже при одинаковом F1 средняя длина K4/V4 и K4/V8
здесь 13 токенов против четырёх у Dynamic/K8/V4/K16/V4: совпадение средних
не означает одинаковую генерацию или одинаковую стоимость.

Все пять политик сохраняются для full. Матрица не выбирается заново
по качеству smoke. Парные CI не переданы и по этим средним не вычислялись.

**Свежие baselines воспроизводят исходную серию.** Во всех восьми
model/dataset/policy source-сводках — ноль отличий output IDs и первых
токенов, delta score=0: **24/24** smoke-пары с expansion_v1 совпали по
сообщению scorer. В обеих задачах все три smoke-документа попали в общую
необрезанную подвыборку; это не устанавливает отсутствие truncation на full.

**Ресурсы.** Из средних длительностей пересчитана оценка основных
1 600 native forward: **2.0371 GPU-часа суммарно**. По job это примерно
26.92 и 25.99 минуты для двух наборов 1.7B, 35.43 и 33.88 для 4B.
Это не wall time и не полный расход Slurm. Есть чувствительность к длинам
трёх документов и ответов; prefill включает обход cache, smoke-контроли
также копируют logits для сверки. Загрузка моделей, GC, дополнительные
reference-генерации и очередь не включены. Full выполняется без trace-копий.
Оставляем прежние четыре job, до двух GPU одновременно, лимит три часа на job.

**Следующий шаг — полная фиксированная матрица.**

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
bash tools/start_mixed_precision.sh full
```

После завершения **всех четырёх** Slurm job:

```bash
python -m src.score_mixed_precision --phase full
cat results/mixed_precision_v1/summary/full/paired_contrasts.csv
cat results/mixed_precision_v1/summary/full/untruncated_contrasts.csv
```

Ожидаются **20 ячеек, 1 600 ответов и 28 основных quality-контрастов**.
HotpotQA sensitivity считается отдельно на общей необрезанной подвыборке.
Код inference, scorer, конфигурация и подготовленные данные после smoke
не менялись; этот snapshot обновляет только документацию и агрегаты.
