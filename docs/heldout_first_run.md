# Независимая выборка LongBench: 80 новых задач на датасет

## Зачем нужна эта серия

В первом пилоте `n=40` на каждом из трёх датасетов INT4 g16
почти сохранил качество DynamicCache, а INT4 g64 ухудшил его.
На `passage_retrieval_en` сокращение ответа с 32 до 8 токенов
повысило балл g64 с 45.23% до 62.50%, но даже короткие ответы
оставались заметно хуже Dynamic и g16 (оба 82.50%).
Первые названные числа во всех 40 случаях для каждого cache
совпали между короткой и длинной сериями.

Это подтвердило необходимость следующего шага: **проверить эффект
на примерах, не участвовавших в первом пилоте**.

## Зафиксированный протокол

- Модель Qwen3-1.7B, FP16, batch size 1, одна V100.
- Три поднабора LongBench: `passage_retrieval_en`,
  `multifieldqa_en`, `multifieldqa_zh`.
- 80 новых задач на датасет, seed 2027, из множества индексов,
  не входящих в исходный пилот `n=40`, seed 2026.
- Динамическое сравнение: `DynamicCache`, `INT4 g16`, `INT4 g64`.
- Промпты, ограничение входа (16 384 токена), HQQ parameters
  и максимум ответа оставляем **как в исходном пилоте**:
  LongBench `dataset2maxlen.json` (32 retrieval, 64 QA).
- Не меняем длину ответа до 8 токенов в этой серии:
  короткий ответ — уже отдельный диагностический контроль.
- Одинаковые token IDs для всех cache; manifest фиксирует
  `source_index` и SHA-256 входа.
- Вариант с 80 задачами **не перезаписывает** оригинальные
  `data/prepared/*.jsonl` и `results/raw/*.jsonl`.

Это 3 датасета × 80 задач × 3 cache = **720 новых генераций**.

## Порядок команд на login-узле ВШЭ

Не выполнять `sbatch` на Mac и не запускать CUDA напрямую на login-узле.

### 1. Обновить репозиторий и проверить unit tests

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
source cluster.env.example
python -m unittest discover -s tests -v
```

Пока ожидается **23 теста**. При ошибках не продолжать GPU-запуск.

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

Должно быть **по 80 записей в held-out** и **по 40 записей в оригинальном пилоте**.
Не запускай `python -m src.prepare_data --n 80`: это изменит
старые подготовленные входы и не гарантирует непересечения.

Если `prepare_heldout` сообщает, что целевая папка уже существует,
не удаляй её: проверь manifest и сообщи об ошибке.

### 3. Запустить сначала только DynamicCache

```bash
mkdir -p logs results/heldout/raw results/heldout/summary
sbatch --array=0,3,6%3 \
  --export=ALL,MODEL_PATH="$MODEL_PATH" \
  slurm/heldout_80.sbatch
```

Сохрани выданный JOBID. Проверки:

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
wc -l results/heldout/raw/*_dynamic.jsonl
```

Ожидаются три файла по 80 ответов.

Можно проверить средние оценки каждого Dynamic датасета, например:

```bash
python -m src.check_smoke results/heldout/raw/passage_retrieval_en_dynamic.jsonl | tail -n 3
python -m src.check_smoke results/heldout/raw/multifieldqa_en_dynamic.jsonl | tail -n 3
python -m src.check_smoke results/heldout/raw/multifieldqa_zh_dynamic.jsonl | tail -n 3
```

Важно: это новая выборка, поэтому baseline может отличаться от
старых чисел. Если ответы выглядят аномально, сначала проверить логи.

### 4. Запустить шесть INT4-заданий после проверки Dynamic

```bash
sbatch --array=1,2,4,5,7,8%3 \
  --export=ALL,MODEL_PATH="$MODEL_PATH" \
  slurm/heldout_80.sbatch
```

Параметр `%3` ограничивает количество одновременно выполняющихся
задач этого массива тремя GPU. Не отправляй полный второй массив,
пока результаты Dynamic не проверены.

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

Сначала сравниваем g16−g64 на 80 новых примерах каждого датасета
и парные bootstrap-интервалы. Нельзя считать 40+80 полностью
независимыми исследованиями после просмотра пилота, но held-out
выборка даёт более сильную проверку воспроизводимости.

Не усредняем три официальные метрики LongBench в один балл.
Различия английского и китайского набора не доказывают
причинный эффект языка.

## Важно о публичном GitHub

Каталоги `data/`, `results/`, `external/` и `plots/` исключены
из коммитов с помощью `.gitignore`.
В публичном репозитории остаются протоколы, код и согласованные
агрегированные результаты, но не веса, архивы и персональные секреты.
