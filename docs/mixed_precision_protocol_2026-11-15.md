**mixed_precision_v1: полная серия завершена 17 ноября 2026 года.**

[Полные результаты](mixed_precision_results_2026-11-17.md): 20 ячеек,
1 600 ответов, 28 основных и 14 subset-контрастов. Ниже я зафиксировал условия
и команды воспроизведения; следующий этап —
[новые контексты](confirmation_plan_2026-11-17.md).

[Результаты smoke](mixed_precision_smoke_2026-11-15.md): 20 ячеек, 60 основных
ответов, stock/exact-сверки и равенство фактических budgets пройдены.
Полная серия также прошла проверку равенства фактических budgets и битности.

[Мотивация и CPU-результаты](kv_probe_diagnostics_2026-11-15.md)
· [Фиксированная матрица](../experiments/mixed_precision_v1.json)
· [Runner](../src/run_mixed_precision.py).

Проверяем практический вопрос: сохраняет ли **K8/V4** native-качество лучше,
чем **K4/V8**, при одинаковом измеренном объёме cache. K16/V4 служит
более дорогим контролем с точными K. Teacher-forced NLL не является
метрикой этой серии: возвращаем обычную свободную greedy-генерацию.

**Матрица.** Qwen3-1.7B и Qwen3-4B, retrieval EN и HotpotQA, все 80
фиксированных expansion_v1 source context на набор. Пять политик:
Dynamic, K4/V4, K8/V4, K4/V8, K16/V4. Всего **20 ячеек, 1 600 ответов**:
960 для новых политик и 640 свежих контролей. Данные и входные token IDs
копируются из проверенных подготовленных expansion-входов без ретокенизации.

У всех условий: context_then_question, native chat без thinking,
FP16/SDPA, HQQ axes=1, group=64, residual=512, ответ до 32 токенов.
Forced-префикса ответа нет. `query + max_new_tokens < residual` проверяется
до запуска; любой дополнительный вызов quantize после первоначальных K,V
останавливает задачу. Новые каталоги prepared/results, старые серии только читаем.

**Почему свежие контроли.** В первоначальном плане предполагалось переиспользовать
640 старых baseline-ответов. Здесь они пересчитываются: все основные контрасты
получают собственные baselines в той же новой реализации и среде. Это также
даёт измерение реальных cache bytes для Dynamic/K4/V4. Прежние ответы сохраняются
для диагностики drift по output IDs и официальному score; они не входят
в знаменатель новых основных контрастов.

**Реализация битности.** Stock HQQ update сохраняется; перед вызовом
оригинального quantize на стороне K/V временно задаётся 4 или 8 bits,
после вызова значение восстанавливается. Фактическая битность проверяется
по HQQ metadata каждой стороны каждого слоя. Для 16-bit стороны quantize
и dequantize обходятся с сохранением точного tensor. На production GPU
такие состояния обязаны быть FP16. Refresh запрещён и отслеживается счётчиками.

Новый ручной split/greedy decoder локально сверен с прежним decoder для
Dynamic и INT4, включая EOS. HQQ 4/8, обход обеих сторон, разные K/V bits,
memory-budget equality и alias-deduplication проверены на маленькой CPU-Qwen.
Это локальная проверка кода в FP32. Совместимость FP16/CUDA на реальных
весах подтверждена присланным полным stdout GPU smoke.

**Что я измеряю как память.** После вопроса, до первого decode-forward:

- `_quantized_keys/_quantized_values`, их tensor metadata scales/zeros;
- точные K/V residual и точные tensors на 16-bit стороне;
- каждый уникальный storage один раз по устройству и адресу, включая
  views, удерживающие более крупный storage. Пустой view не исключает
  удерживаемый непустой storage из суммы.

Пишем GPU bytes, host bytes, число storages, фактические bits и число
cache tokens. Веса модели, allocator reserve и временные dequantized tensors
в эту resident-cache величину не входят. Это не peak GPU memory.
K8/V4 и K4/V8 должны иметь **равные GPU bytes на каждом входе**, а не только
в среднем; иначе scorer останавливается до вывода о равном бюджете.
Измерение выполнено в общей точке до расхождения длин свободных ответов.

Номинальный payload K8/V4 и K4/V8 — 37.5% от FP16 KV при одинаковом числе
элементов K/V; K4/V4 — 25%, K16/V4 — 62.5%. Это только арифметика битности:
metadata/residual учитываются фактическим измерением, а проценты не являются
экономией всей GPU RAM.

**Smoke.** Первые три прежних документа на набор, те же пять политик:
**60 основных ответов** плюс **48 reference-генераций**:

- 12 обходов обеих сторон K16/V16 против основных Dynamic;
- 12 исходных stock INT4 против основных K4/V4;
- 12 stock INT8 и 12 смешанных K8/V8 для прямой сверки.

Проверяются все greedy output IDs и все next-token logits по траектории,
не только первый токен. Source/manifest hashes, версии, GPU/CUDA между
smoke и full должны совпасть. Локальные unit-тесты не заменяют этот этап.
Full helper проверяет весь smoke и равенство memory budgets до отправки.

**Команды.** На login-узле:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
bash tools/start_mixed_precision.sh smoke
```

Helper готовит оба model manifest, проверяет все четыре model/dataset
входа и отправляет четыре Slurm job: по одному на модель и набор,
до двух GPU одновременно. Модель загружается один раз на job; в нём
последовательно выполняются пять политик. Пакеты и веса обновлять не нужно.
После завершения всех четырёх job:

```bash
python -m src.score_mixed_precision --phase smoke
```

Нужно получить **20 ячеек, 60 ответов**, успешные stock/exact-сверки,
ожидаемые bits, отсутствие refresh и равенство GPU cache bytes K8/V4–K4/V8
на всех задачах. Качество на трёх документах не используем для выбора политики.
Оценка времени full по smoke грубая; 0.282 GPU-часа прошлой teacher-forced
серии не переносим на native decode.

Только после успешного smoke:

```bash
bash tools/start_mixed_precision.sh full
# После завершения:
python -m src.score_mixed_precision --phase full
cat results/mixed_precision_v1/summary/full/paired_contrasts.csv
```

Full: 1 600 основных ответов, smoke не добавляется к n=80. Итоговые
`dataset_scores.csv`, `memory_budget.csv`, `reference_drift.csv`,
`paired_contrasts.csv`, `untruncated_counts.csv`, `untruncated_contrasts.csv`.
Основных quality-контрастов 28; HotpotQA sensitivity считается на общей
необрезанной подвыборке двух Qwen, размер выводится scorer.
Primary — K8/V4−K4/V8; дополнительные — сравнения с K4/V4 и свежим Dynamic.
Интервалы парные по context, 2 000 bootstrap, seed 2026, номинальные 95%,
exploratory, без поправки множественных сравнений.

**Ресурсы.** Один GPU и четыре CPU на job, лимит три часа на job,
массив 0–3%2. 12 GPU-часов — предельное выделенное время четырёх job,
не ожидаемое потребление. Длительности prefill включают snapshot-обход
cache; различающиеся длины свободных ответов не дают чистого сравнения
throughput. Загрузка модели, GC, reference-сверки и очередь не входят
в выводимую сумму основных forward. Частичный совместимый запуск можно
возобновить той же командой. Логи `logs/lb-mixed-v1-<array>_<index>.err/.out`.

Серия проверяет исправление на уже изученных данных. Выбор лучшей политики
после этих результатов потребует последующего независимого подтверждения;
native-качество, условный NLL и идеальные проценты payload не объединяем
в один показатель.
