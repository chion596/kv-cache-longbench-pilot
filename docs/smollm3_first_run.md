# Смежная архитектура без одобрения Meta: SmolLM3-3B

## 26 октября: QA-smoke завершён, готовы 80 QA-заданий

Массив Slurm **4446597** завершён: три раза
`COMPLETED 0:0`, получено 24 ответа.
На восьми первых `multifieldqa_en` средний
официальный QA F1, рассчитанный из округлённых баллов:
Dynamic **56.29%**, g16 **45.89%**, g64 **48.76%**.

Главный пример: при эталоне `Vice Admiral`
первый ответ верный у всех cache, но g16/g64 добавляют
много лишнего текста, поэтому F1 падает
со 100% до 10.8% и 7.7%.
У восьми заданий нет статистической силы
для вывода об общей деградации QA.

[Отчёт и команда полного QA-прогона](smollm3_qa_smoke_results.md).
Добавлены `src.diagnose_family_qa` для сравнения
F1 исходного ответа и первой строки без GPU и
`slurm/smollm3_qa_en80.sbatch` для 80 заданий × 3 cache.

## Диагностика завершена, QA-smoke отправлен: Slurm 4446597

Команда `python -m unittest discover -s tests -v`:
**44 tests, OK**.

Команда `python -m src.diagnose_family_retrieval`:
- Qwen3: при Dynamic/g64/g128 первое число совпало с эталоном
  соответственно в **66/42/37 из 80** случаев. Все Qwen3 ответы
  достигли лимита 32 токена.
- SmolLM3: правильное первое число **75/80 при всех пяти cache**;
  предел генерации достигнут в 6–8 случаях, большинство ответов
  короткие.
- SmolLM3 g64 снизил официальный score в одном задании, g128 —
  в двух, при сохранении верного первого номера.
- Qwen3 медиана peak Dynamic **5.223 GiB**; SmolLM3 **7.663 GiB**.
  Средняя парная экономия peak allocation при g64:
  **18.884%** для Qwen3 и **8.037%** для SmolLM3.

[Подробные данные и интерпретация](smollm3_retrieval80_results.md).

**Новый GPU-запуск:** smoke на первых восьми
`multifieldqa_en` заданиях, массив **4446597**, три cache
Dynamic/g16/g64, всего предполагается 24 ответа.
На момент записи известно только, что задание
**поставлено в очередь** (`Submitted batch job 4446597`).
Не считать его завершённым без результатов `sacct`.

После работы Slurm:

```bash
cd ~/kv-cache-longbench-pilot
sacct -j 4446597 --format=JobID,State,ExitCode,Elapsed
wc -l results/families/smollm3-3b/qa_en_smoke/*.jsonl
python -m src.check_smoke \
  results/families/smollm3-3b/qa_en_smoke/*.jsonl
```

Проверка должна подтвердить три файла по восемь ответов,
а затем — осмысленность baseline Dynamic и возможность
сравнивать F1 с квантованными cache. Маленькая последовательная
подвыборка из восьми заданий не заменяет полной QA-оценки.

## Статус на 25 октября: все пять групп завершены

Полный результат на `passage_retrieval_en`: 80 одинаковых исходных
заданий × Dynamic, g16, g32, g64 и g128 = **400 генераций**.
Официальные баллы соответственно **93.750 / 93.750 / 92.500 /
93.125 / 92.500%**. Разности с Dynamic в SmolLM3
невелики и их парные bootstrap-интервалы включают ноль.
На Qwen3-1.7B в той же выборке оценки резко падали при g64/g128.

[Полный отчёт, протокол и ограничения](smollm3_retrieval80_results.md).

Следующий эксперимент не повторяет retrieval: проверяем
переносимость на **`multifieldqa_en`**, поскольку короткий
числовой ответ retrieval может скрывать деградацию текста.

### Быстрая диагностика без GPU

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
source cluster.env.example
python -m unittest discover -s tests -v
python -m src.diagnose_family_retrieval
```

Диагностика сравнивает Qwen3 и SmolLM3 на совпадающих
`source_index`, но НЕ требует совпадения token IDs между
моделями. Она показывает первое число, лишние числа в ответе,
достижение лимита, ошибки отдельных задач и экономию
`peak_gpu_gib` относительно Dynamic в каждой модели.
Это позволяет уточнить, чем объясняется разница официального
`retrieval_score`.

### Следующий GPU-запуск: QA sanity check, 8 задач × 3 cache

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
source cluster.env.example
export OTHER_MODEL_PATH="$(ws_find kv-cache)/models/SmolLM3-3B"
wc -l data/prepared/families/smollm3-3b/multifieldqa_en.jsonl
mkdir -p logs results/families/smollm3-3b/qa_en_smoke
sbatch --export=ALL,OTHER_MODEL_PATH="$OTHER_MODEL_PATH" \
  slurm/smollm3_qa_en_smoke.sbatch
```

Ожидаются **80 подготовленных** QA-строк и **три по 8**
записей после успешного завершения Slurm array.

Затем на login-узле:

```bash
wc -l results/families/smollm3-3b/qa_en_smoke/*.jsonl
python -m src.check_smoke \
  results/families/smollm3-3b/qa_en_smoke/*.jsonl
```

Это **не статистический результат**: всего восемь первых
QA-заданий, не отдельная случайная выборка. Цель — изучить
правдоподобие Dynamic baseline и ответы raw LongBench,
до запуска 80 задач. Скрипт `src.score_family_qa` уже
подготовлен для последующей полноценной QA-матрицы.
Пока не стоит выполнять её до анализа smoke.

## Почему изменили порядок

При попытке загрузить `meta-llama/Llama-3.2-1B-Instruct`
Hugging Face подтвердил вход в аккаунт, но отклонил скачивание:
`Access denied. This repository requires approval.`
Это ограничение модели на стороне Hugging Face/Meta, а не ошибка
`hf auth login`, CUDA или Slurm.

Одобрение Llama можно запросить в браузере:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
Описание gated access:
https://huggingface.co/docs/hub/models-gated

Пока одобрение не получено, отдельно проверяем
`HuggingFaceTB/SmolLM3-3B`, модель другого семейства
со свободно доступными весами и контекстом до 65 536 токенов
в стандартном config. Весит около 3 млрд параметров, поэтому
её нельзя считать контролем влияния **только архитектуры**
при сравнении с Qwen3-1.7B.

Страница модели: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

## Журнал: 22 октября 2026 года

Пользователь отправил проверочный Slurm-массив **4433864**
для SmolLM3-3B. Три задачи массива успешно завершились по данным `sacct`:
`4433864_0` Dynamic (`00:06:44`),
`4433864_1` INT4 g16 (`00:04:37`),
`4433864_2` INT4 g64 (`00:04:14`).
Для каждой задачи статус `COMPLETED`, код `0:0`.
В каталоге `results/families/smollm3-3b/smoke/`
есть три JSONL по три ответа, всего **9 записей**.

**Вывод на данном этапе:** техническая совместимость
SmolLM3-3B с тремя cache-конфигурациями подтверждена,
однако содержимое ответов и их официальные баллы
ещё не анализировались. Три задачи — это smoke test,
а не свидетельство улучшения/деградации качества модели.

**Smoke test по официальному `retrieval_score` также прошёл.**
Все три конфигурации правильно ответили на все три задания:

| Задание | Эталон | Dynamic | INT4 g16 | INT4 g64 |
|---|---|---|---|---|
| passage_retrieval_en:5 | Paragraph 3 | `3.` (100%) | `3.` (100%) | `3` (100%) |
| passage_retrieval_en:6 | Paragraph 24 | `24.` (100%) | `24.` (100%) | `24.` (100%) |
| passage_retrieval_en:9 | Paragraph 12 | `12.` (100%) | `12.` (100%) | `12.` (100%) |

Длины входов: 12 103, 11 767 и 14 096 токенов. В отличие
от наблюдавшихся на Qwen3-1.7B длинных повторяющихся продолжений,
в этих трёх заданиях SmolLM3-3B выдаёт краткие ответы.
При этом **3/3 недостаточно**, чтобы делать вывод об устойчивости
к group size или об абсолютном качестве модели на LongBench.

Следующая серия: по 80 **тех же исходных задач** `passage_retrieval_en`
для Dynamic/g16/g64. Не смешивать эти 240 генераций со smoke-файлами,
сохранёнными отдельно. Готовы `slurm/smollm3_retrieval80.sbatch`
и `src/score_family_retrieval.py`, который проверит идентичность
входов, посчитает парные разницы с интервалами и длину генерации.

## Результаты полного retrieval-прогона (24 октября 2026)

Три файла по 80 ответов получены, проверка
`python -m src.score_family_retrieval` выполнена успешно:

- Dynamic: **93.750%**, 5/80 нулевых, 7/80 достигли лимита,
  средняя длина 5.8 токенов.
- INT4 g16: **93.750%**, 5/80 нулевых, 6/80 достигли лимита,
  средняя длина 5.2 токена.
- INT4 g64: **93.125%**, 5/80 нулевых, 8/80 достигли лимита,
  средняя длина 6.5 токенов.
- Разница g64 − g16: **−0.625 п.п.**,
  95% парный bootstrap **[−1.875; 0.000]** п.п.

Поэтому ожидавшаяся по Qwen3-1.7B серьёзная деградация
при g64 на SmolLM3-3B **не проявилась**.
Это сильное различие в наблюдаемом поведении моделей,
но не идентификация влияния одной архитектуры
(разные размеры, обучение и токенизация).

См. [полный разбор результатов](smollm3_retrieval80_results.md).

Чтобы достроить group-size curve без повторения уже завершённых
240 генераций, запускаем **только** индексы g32/g128:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
source cluster.env.example
export OTHER_MODEL_PATH="$(ws_find kv-cache)/models/SmolLM3-3B"
mkdir -p logs results/families/smollm3-3b/retrieval80
sbatch --array=2,4%2 \
  --export=ALL,OTHER_MODEL_PATH="$OTHER_MODEL_PATH" \
  slurm/smollm3_retrieval80.sbatch
```

После завершения:

```bash
wc -l results/families/smollm3-3b/retrieval80/*.jsonl
python -m src.score_family_retrieval \
  --caches dynamic int4_g16 int4_g32 int4_g64 int4_g128
```

## Следующий запуск: 80 retrieval-заданий

В login shell кластера:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
source cluster.env.example
export OTHER_MODEL_PATH="$(ws_find kv-cache)/models/SmolLM3-3B"
python -m unittest discover -s tests -v
wc -l data/prepared/families/smollm3-3b/passage_retrieval_en.jsonl
```

Если unit tests завершились `OK` и подготовлено ровно 80
заданий, запускать первую серию:

```bash
mkdir -p logs results/families/smollm3-3b/retrieval80
sbatch --array=0,1,3%2 --export=ALL,OTHER_MODEL_PATH="$OTHER_MODEL_PATH" \
  slurm/smollm3_retrieval80.sbatch
```

Массив использует нумерацию `0=Dynamic`, `1=g16`,
`2=g32`, `3=g64`, `4=g128`, но первый запуск включает
**только** `0,1,3`. Максимум две одновременно используемые GPU.

После завершения всех трёх Slurm-задач (`COMPLETED 0:0`):

```bash
wc -l results/families/smollm3-3b/retrieval80/*.jsonl
python -m src.score_family_retrieval
```

Ожидаются три файла по 80 ответов. Официальная метрика
сравнивается попарно на одинаковых задачах. Скрипт напечатает
качество, долю нулевых баллов, частоту достижения предела длины
и парный bootstrap-интервал для g16−g64.

**Решение по расширению:** если результаты информативны,
на тех же входах добавим g32 и g128 (`--array=2,4%2`),
а затем отдельно проверим `multifieldqa_en`. Китайский QA
не стоит запускать большой серией без sanity-check baseline.


Параллельно статус запроса доступа к Meta Llama 3.2 на Hugging Face:
**PENDING**. Это значит, что разрешения на скачивание весов Llama
пока нет; вход в Hugging Face ранее уже прошёл успешно.

Для проверки:

```bash
sacct -j 4433864 --format=JobID,State,ExitCode,Elapsed
wc -l results/families/smollm3-3b/smoke/*.jsonl
```

Если любая из трёх задач завершилась ошибкой, нужно сначала
посмотреть её `logs/lb-smollm3-smoke-4433864_*.err`, а не
запускать большую матрицу.

## Первый этап — только проверка, а не полноценный бенчмарк

В `src.prepare_family_model` уже есть подготовка **тех же 80 задач**
по `source_index`, что в Qwen3 held-out. При этом модель
получает собственную токенизацию, как и должна.

Первый Slurm array включает лишь три конфигурации:
- DynamicCache;
- HQQ INT4 `q_group_size=16`;
- HQQ INT4 `q_group_size=64`.

По три первых задания английского retrieval; всего 9 ответов.
Цель — проверить совместимость API Transformers, attention/cache,
запуск на V100 и формат вывода. Статистической силы у этих 9
ответов **нет**.

## Команды на login-узле ВШЭ

Обновить код и проверить тесты:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
source cluster.env.example
python -m unittest discover -s tests -v
```

Скачать веса в **отдельную директорию** уже продлённого workspace:

```bash
export OTHER_MODEL_PATH="$(ws_find kv-cache)/models/SmolLM3-3B"
hf download HuggingFaceTB/SmolLM3-3B --local-dir "$OTHER_MODEL_PATH"
ls -lh "$OTHER_MODEL_PATH/config.json"
```

Отказ Meta в доступе к Llama не мешает загрузке этого
открытого репозитория. Не удалять и не переиспользовать
незавершённую директорию Llama.

Подготовить входы **без GPU**, используя LongBench из
предыдущей серии:

```bash
python -m src.prepare_family_model \
  --model "$OTHER_MODEL_PATH" \
  --name smollm3-3b

wc -l data/prepared/families/smollm3-3b/*.jsonl
```

Ожидаем три файла по 80 заданий. Сохраняем manifest:
там зафиксированы индексы, tokenizer, входные token hashes,
длина и число усечений. Старые Qwen-файлы остаются нетронутыми.

Проверить совместимость на трёх первых retrieval-заданиях:

```bash
mkdir -p logs results/families/smollm3-3b/smoke

sbatch --export=ALL,OTHER_MODEL_PATH="$OTHER_MODEL_PATH" \
  slurm/smollm3_family_smoke.sbatch
```

После запуска подставить реальный `JOBID`:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
wc -l results/families/smollm3-3b/smoke/*.jsonl
```

Ожидаемые три файла по три строки и три `COMPLETED 0:0`.
Если возникает CUDA OOM, ошибка HQQ, позиционирования attention
или обработчика `logits_to_keep`, изучить `logs/lb-smollm3-smoke-...`
и остановиться: сначала исправить совместимость, **не**
запускать 80 задач для всех cache.

## Методологические ограничения

- SmolLM3 — 3B, Qwen3 — 1.7B: size, training и architecture
  отличаются одновременно; сравнение только проверяет
  **переносимость** эффекта, не чистый эффект architecture family.
- SmolLM3 поддерживает extended thinking, что может сильно
  влиять на raw completion. Наш smoke сохраняет исходный протокол
  **без chat template**. В случае слабого Dynamic необходимо
  отдельно оценить поведение с фиксированным шаблоном, не менять
  его посередине серии.
- Для китайских вопросов качество SmolLM3 может быть ниже:
  по официальной карточке заявлены шесть поддерживаемых языков
  (без китайского). До полной серии сначала проверить
  информативность baseline на каждом датасете.
- Сравнение памяти/latency между моделями требует отдельного
  контроля размеров, token length и одинакового hardware.
