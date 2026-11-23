**ZH confirmation: протокол зафиксирован 18 ноября 2026 года.**

**Обновление 21 ноября:** smoke завершён — 8 ячеек / 24 ответа; проверки
битности, refresh, stock/exact parity и равной primary-памяти прошли.
Качество трёх stress-примеров не обобщаем. [Отчёт smoke](zh_confirmation_smoke_2026-11-21.md).
Следующий этап — отдельный full запуск на 120 документах; он пока не выполнялся.

CPU-preview подтвердил доступность 120 новых exact context, но один вопрос
занимает 815 токенов. С лимитом ответа 32 получаем `815 + 32 = 847`;
условию без refresh нужен residual **строго больше 847**. Для отдельной
серии `zh_confirmation_v1` выбран общий **residual=1024**. Ни один случай
не исключается, seed и membership прежнего preview сохраняются.

[Матрица](../experiments/zh_confirmation_v1.json)
· [Полученный CPU-результат](zh_preflight_results_2026-11-18.md)
· [Исходный план](confirmation_plan_2026-11-17.md).

**Гипотеза и условия.** На новых китайских retrieval-документах K8/V4
лучше K4/V8 при равном фактическом resident KV cache. Это проверяется
отдельно на Qwen3-1.7B и Qwen3-4B. Официальный `retrieval_zh_score` —
основная метрика; обе политики генерируют свободный native-ответ.

| Ось | Зафиксировано |
|---|---|
| Документы | 120 из прежнего CPU-preview, один вопрос на exact context |
| Набор | passage_retrieval_zh, официальный pinned LongBench archive |
| Sampling | uniform exact context, seed=2043, без отбора по исходам или длине |
| Модели | Qwen3-1.7B, Qwen3-4B; те же локальные artifacts/версии, что в mixed_precision_v1 |
| Политики | Dynamic, K4/V4, K8/V4, K4/V8 |
| Prompt / prefill | официальный ZH, native chat, thinking=False; context_then_question |
| Числа | FP16, SDPA, stock HQQ update, axes K/V=1, group=64, residual=1024 |
| Вход / ответ | максимум 16 384 / 32 токена, greedy, без forced prefix |
| Объём | 8 ячеек × 120 = **960 основных ответов**, 120 разных документов |
| Сравнения | все controls считаются заново на тех же входах и в той же среде |

Распределение бит и cache instrumentation переиспользованы из завершённой
mixed_precision_v1; её inference-файлы и fingerprint не изменены.
В новой серии измеряются реальные storages после query, перед первым
decode-шагом: cache, metadata, точный residual, без весов/allocator reserve/
transient dequant buffers. Равенство K8/V4–K4/V8 GPU bytes проверяется
**на каждом входе**, а не только по средним. Битность и ровно одна начальная
квантизация каждого K/V во всех слоях обязательны; refresh вызывает ошибку.

**Независимость и фиксация.** Подготовка читает существующий кластерный JSON
`results/zh_confirmation_preflight_2026-11-17.json`, проверяет hash архива,
официальные файлы, parent manifests, tokenizer, версии и повторяет history scan.
Если список доступных документов изменился, она останавливается, сохраняя
прежние файлы; автоматически другую выборку не выбирает. Prepared-входы всех
политик/моделей и selection записываются одной новой immutable серией.
Повторная подготовка только проверяет сохранённые входы. После фиксации
выборки её случаи уже видны в истории; старый preview я повторно не запускаю.

История покрывает опубликованный аудит и локальные `data/prepared`, `results`.
Exact hashes не исключают смысловые дубли, переводы, pretraining contamination
или запуски, которых нет в этих источниках. Сами индексы и hashes preview
остались на кластере; stdout без них не заменяет selection manifest.

**Статистика до ZH-генераций.** Primary family — два контраста
`K8/V4 − K4/V8`, по одному на модель, официальный score в п.п.
Парный percentile bootstrap по 120 документам, 10 000 повторов, seed=2043.
Для этих двух контрастов используются **97.5% интервалы**: поправка Bonferroni
даёт приблизительное семейное покрытие 95% при валидном bootstrap.
Scorer дополнительно сохраняет nominal 95% для сопоставимости.
Нижняя граница primary CI выше нуля поддерживает преимущество на данной
модели; перенос на обе Qwen требует этого у обеих, без последующего выбора
удачной модели. Это приближённые bootstrap-интервалы, не точный finite-sample тест.

Secondary: K8/V4−Dynamic, K4/V8−Dynamic, K4/V4−Dynamic, K8/V4−K4/V4;
nominal 95%, exploratory, без коррекции. Дополнительно: EOS, первый номер,
изменения score и output IDs по случаям, truncation, реальная память и
длительности forward. Все 120 документов участвуют независимо от правильности
Dynamic; никаких фильтров по качеству baseline. Первое число — вспомогательная
метрика, не замена официальной. Эквивалентность Dynamic и non-inferiority
не заявляются без заранее обоснованного допустимого ухудшения.

**Шаг 1: только smoke.** На login-узле:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
bash tools/start_zh_confirmation.sh smoke
```

Сначала CPU-подготовка и проверки; затем два Slurm job, по одному на модель,
четыре политики внутри каждого. По 1 GPU / 4 CPU, не более двух job одновременно,
лимит 3 часа на job. Это верхний лимит 6 GPU-часов на фазу, не прогноз расхода.
Сеть отключена, используются прежние локальные модели; git на GPU-узле не нужен.
Пути можно задать через `KV_MODEL_ROOT`, `QWEN17_MODEL_PATH`, `QWEN4_MODEL_PATH`;
иной путь preview — через `ZH_PREFLIGHT_JSON`.

Общие smoke-cases выбираются **до генерации по длинам**: самый длинный query,
самый короткий и самый длинный input у обеих tokenizers; совпадения дополняются
по source index. Случай с query815 обязательно входит. Всего **24 основных
ответа + 24 reference-генерации** для stock HQQ 4/8 и exact/Dynamic сверок
logits/output IDs. Качество и runtime по этим трём stress-cases не обобщаются.

После завершения обоих job:

```bash
python -m src.score_zh_confirmation --phase smoke
```

Я ожидаю 8 ячеек / 24 ответа, пройденные stock/exact сверки, отсутствие refresh
и равные primary bytes. При падении сначала смотрю `logs/lb-zh-v1-*.err`
и `sacct -j ARRAY_ID --format=JobID,State,ExitCode,Elapsed`; full пока не запускаю.

**Шаг 2: full после разбора успешного smoke.**

```bash
bash tools/start_zh_confirmation.sh full
```

Runner перед отправкой и на GPU проверяет успешную **всю** smoke-матрицу;
source hashes, manifest/model/versions, GPU type и CUDA должны совпасть.
Full не отправляется автоматически; smoke-результаты не дописываются в full.
Завершённые задачи при resume не перезаписываются, смешивать fingerprints нельзя.

После обоих job:

```bash
python -m src.score_zh_confirmation --phase full
cat results/zh_confirmation_v1/summary/full/paired_contrasts.csv
```

Ожидаются 960 ответов, 8 scores, 10 контрастов (2 primary + 8 secondary),
2 проверки памяти и 960 поэлементных строк. Scores и CI вычисляются только
после проверки полной сопоставимой матрицы. Forward time исключает загрузку,
GC, очередь и reference checks; ускорение/throughput этим не доказываются.

**Граница вывода.** Это перенос выбранной политики на новые ZH-документы
при residual1024. EN использовала другие документы и residual512; прямую
разность нельзя считать причинным эффектом языка. Больший residual сохраняет
больше точных query-состояний, поэтому прежние 41% cache Dynamic на ZH
заранее не обещаются: память измеряется заново. Для изоляции языка понадобятся
согласованные residual/длины и сопоставимые или параллельные тексты в новой серии.
