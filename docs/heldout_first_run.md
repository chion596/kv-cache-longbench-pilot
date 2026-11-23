# Независимая выборка LongBench: 80 новых задач на датасет

## Зачем нужна эта серия

В первом пилоте `n=40` на каждом из трёх датасетов INT4 g16
почти сохранил качество DynamicCache, а INT4 g64 ухудшил его.
На `passage_retrieval_en` сокращение ответа с 32 до 8 токенов
повысило балл g64 с 45.23% до 62.50%, но даже короткие ответы
оставались заметно хуже Dynamic и g16 (оба 82.50%).
Первые названные числа во всех 40 случаях для каждого cache
совпали между короткой и длинной сериями.

Поэтому следующим шагом я решил проверить эффект **на примерах,
не участвовавших в первом пилоте**.

## Журнал выполнения — 22 октября 2026 года

По выводу с login-узла кластера:

- 23 локальных unit-теста завершились успешно (`OK`).
- `src.prepare_heldout` создал по 80 новых заданий на каждый датасет,
  исключив исходные 40 задач.
- Подготовленные входы: `passage_retrieval_en` — 0 усечений,
  `multifieldqa_en` — **1 усечение**, `multifieldqa_zh` — 0.
- Оригинальные три файла по 40 заданий сохранены без изменений.
- **DynamicCache array отправлен в Slurm**: Job ID `4432728`,
  индексы `0,3,6`, максимум 3 параллельных задачи.
- На новой выборке получено **по 80 ответов DynamicCache** для всех трёх датасетов,
  всего 240 записей. Проверка статуса Slurm через `sacct` пока не предоставлена.
- Средний балл DynamicCache на новых заданиях (округление до десятых):
  `passage_retrieval_en` — **80.6%**,
  `multifieldqa_en` — **24.6%**,
  `multifieldqa_zh` — **20.9%**.
- Первые 40 задач показывали соответственно **80.42%**, **27.62%**
  и **23.49%**. На новых задачах baseline остался сопоставимым;
  различия двух выборок не являются мерой влияния квантизации.
- Эксперименты `INT4 g16/g64` на новых заданиях пока **не запускались
  или не подтверждены результатами**; их нужно выполнить на тех же
  входных токенах из `data/prepared/heldout`.

При анализе учесть усечённое задание: внутри held-out серии оно должно
быть идентично для всех cache; дополнительно можно провести
sensitivity analysis без него, не меняя исходный протокол или файлы.

## Зафиксированный протокол

- Модель Qwen3-1.7B, FP16, batch size 1, одна V100.
- Три поднабора LongBench: `passage_retrieval_en`,
  `multifieldqa_en`, `multifieldqa_zh`.
- 80 новых задач на датасет, seed 2027, из множества индексов,
  не входящих в исходный пилот `n=40`, seed 2026.
- Динамическое сравнение: `DynamicCache`, `INT4 g16`, `INT4 g64`.
- Промпты, ограничение входа (16 384 токена), HQQ parameters
  и максимум ответа я оставил **как в исходном пилоте**:
  LongBench `dataset2maxlen.json` (32 retrieval, 64 QA).
- Не меняем длину ответа до 8 токенов в этой серии:
  короткий ответ — уже отдельный диагностический контроль.
- Одинаковые token IDs для всех cache; manifest фиксирует
  `source_index` и SHA-256 входа.
- Вариант с 80 задачами **не перезаписывает** оригинальные
  `data/prepared/*.jsonl` и `results/raw/*.jsonl`.

Это 3 датасета × 80 задач × 3 cache = **720 новых генераций**.

## Порядок команд на login-узле ВШЭ

Я не выполняю `sbatch` на Mac и не запускаю CUDA напрямую на login-узле.

### 1. Обновить репозиторий и проверить unit tests

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
source cluster.env.example
python -m unittest discover -s tests -v
```

На тот момент я ожидал **23 теста**. При ошибках я не продолжал GPU-запуск.

### 2. Зафиксировать и подготовить 80 новых заданий

```bash
python -m src.prepare_heldout --model "$MODEL_PATH"
```

Код проверяет, что:

- исходный пилот имел seed=2026 и n=40;
- выбранные индексы соответствуют пилоту;
- нет пересечений между исходной и новой выборкой;
- архив LongBench, commit upstream и токенизатор соответствуют исходной серии;
- каталог `data/prepared/heldout` ещё не существует.

Далее:

```bash
wc -l data/prepared/heldout/*.jsonl
cat data/prepared/heldout/manifest.json | head -n 25
wc -l data/prepared/*.jsonl
```

Я проверяю, что в held-out **по 80 записей**, а в оригинальном пилоте **по 40**.
Я не запускаю `python -m src.prepare_data --n 80`: эта команда изменит
старые подготовленные входы и не гарантирует непересечения.

Если `prepare_heldout` сообщает, что целевая папка уже существует,
я не удаляю её, а сначала проверяю manifest и разбираю ошибку.

### 3. Запустить сначала только DynamicCache

```bash
mkdir -p logs results/heldout/raw results/heldout/summary
sbatch --array=0,3,6%3 \
  --export=ALL,MODEL_PATH="$MODEL_PATH" \
  slurm/heldout_80.sbatch
```

Выданный JOBID я сохраняю и проверяю:

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
wc -l results/heldout/raw/*_dynamic.jsonl
```

Ожидаются три файла по 80 ответов.

Средние оценки Dynamic по каждому датасету я проверяю, например, так:

```bash
python -m src.check_smoke results/heldout/raw/passage_retrieval_en_dynamic.jsonl | tail -n 3
python -m src.check_smoke results/heldout/raw/multifieldqa_en_dynamic.jsonl | tail -n 3
python -m src.check_smoke results/heldout/raw/multifieldqa_zh_dynamic.jsonl | tail -n 3
```

Это новая выборка, поэтому baseline может отличаться от старых чисел.
Если ответы выглядят аномально, я сначала проверяю логи.

### 4. Запустить шесть INT4-заданий после проверки Dynamic

```bash
sbatch --array=1,2,4,5,7,8%3 \
  --export=ALL,MODEL_PATH="$MODEL_PATH" \
  slurm/heldout_80.sbatch
```

Параметр `%3` ограничивает количество одновременно выполняющихся
задач этого массива тремя GPU. Полный второй массив я отправляю только
после проверки результатов Dynamic.

### 5. Анализ после завершения всех девяти заданий

```bash
wc -l results/heldout/raw/*.jsonl
python -m src.score \
  --results-dir results/heldout/raw \
  --out-dir results/heldout/summary \
  --longbench-code external/LongBench/LongBench

python -m src.diagnose_pilot \
  --results-dir results/heldout/raw \
  --expected-n 80 \
  --examples 2

python -m src.plot \
  --input results/heldout/summary/dataset_scores.csv \
  --out plots/heldout_80_scores.png
```

Ожидаются **9 файлов по 80 строк**, всего 720 ответов.
`src.score` останавливается при неполной матрице или при
несовпадении `task_id`/хешей входных токенов между cache.

## Как интерпретировать

Сначала я сравниваю g16−g64 на 80 новых примерах каждого датасета
и парные bootstrap-интервалы. Я не считаю 40+80 полностью
независимыми исследованиями после просмотра пилота, но held-out
выборка даёт более сильную проверку воспроизводимости.

Я не усредняю три официальные метрики LongBench в один балл.
Различия английского и китайского наборов не доказывают
причинный эффект языка.

## Важно о публичном GitHub

Каталоги `data/`, `results/`, `external/` и `plots/` исключены
из коммитов с помощью `.gitignore`.
В публичном репозитории остаются протоколы, код и согласованные
агрегированные результаты, но не веса, архивы и персональные секреты.
