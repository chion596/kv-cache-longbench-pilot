# Следующая серия LongBench: g32 и g128

## Зачем

На двух непересекающихся выборках из 40 и 80 заданий
`INT4 g16` оказался существенно устойчивее `INT4 g64`.
Теперь проверяем, как меняется качество между ними и после g64.
Это **новая конфигурационная серия на уже выбранных 80 заданиях**,
а не независимая третья выборка.

Полные результаты предыдущих двух запусков:
[первый пилот](pilot_results_2026-10-22.md) и
[проверка на новых 80 заданиях](heldout_80_results.md).

## Дизайн

- Модель: Qwen3-1.7B, одна V100, FP16.
- Данные: **те же** 80 непересекающихся с первым пилотом задач
  каждого из трёх датасетов.
- Старые результаты: Dynamic, g16, g64 — не перезаписываются.
- Новые варианты: HQQ INT4 `g32` и `g128`.
- HQQ axis 1/1, residual 128, raw LongBench prompt и лимиты
  генерации прежние.
- 6 GPU-заданий × 80 ответов = **480 новых генераций**.
- Не более трёх заданий одновременно (`%3`).

Гипотеза: на Qwen3-1.7B рост `q_group_size` от 16 к 128
приведёт к ухудшению официальных метрик хотя бы на части задач.
Однако возможна **немонотонная зависимость**, её нельзя исключать
по прошлому сравнению всего двух групп.

## Команды на login-узле ВШЭ

### 1. Обновление и тесты

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
source cluster.env.example
python -m unittest discover -s tests -v
```

Ожидается **28 тестов**. Если какой-то тест не прошёл — остановиться.

### 2. Сохранность существующей серии

```bash
wc -l data/prepared/heldout/*.jsonl
wc -l results/heldout/raw/*.jsonl
```

Перед отправкой новых заданий ожидаем три prepared-файла
по 80 строк и девять raw-файлов по 80 строк.

### 3. Отправить только g32/g128

```bash
mkdir -p logs results/heldout/raw
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" \
  slurm/group_size_heldout.sbatch
```

Выданный Job ID запиши. Массив `0–5`:
- 0/1: passage_retrieval_en, g32/g128;
- 2/3: multifieldqa_en, g32/g128;
- 4/5: multifieldqa_zh, g32/g128.

Отслеживание:

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Дождись шести `COMPLETED 0:0` прежде, чем считать матрицу.

### 4. Анализ без GPU

```bash
wc -l results/heldout/raw/*.jsonl

python -m src.score_group_ablation \
  --results-dir results/heldout/raw \
  --out-dir results/heldout/group_size

cat results/heldout/group_size/dataset_scores.csv
cat results/heldout/group_size/paired_comparisons.csv

python -m src.plot_group_ablation
```

Ожидаются **15 файлов по 80 ответов**:
`3 датасета × (Dynamic, g16, g32, g64, g128)`.

Скрипт проверяет количество ответов, равенство `task_id`,
совпадение хешей входных токенов, модели, ограничений генерации
и HQQ-параметров. Он не смешивает эти данные с предыдущими
результатами `n=40` и не изменяет код официальных метрик.

Итоговые файлы:
- `results/heldout/group_size/dataset_scores.csv`;
- `results/heldout/group_size/paired_comparisons.csv`;
- `results/heldout/group_size/example_scores.csv`;
- `plots/heldout_group_size.png`.

## Что анализировать

В первую очередь кривые качества по g16/g32/g64/g128
**отдельно для каждого датасета** и прямые парные контрасты.

Если качество при g32 оказывается уже существенно ниже g16,
чувствительность может проявляться раньше g64.
Если g128 почти не отличается от g64, возможно, мы упёрлись
в «пол» качества.

Поскольку почти все исходные ответы достигали ограничений
32/64 токена, учитывать возможность деградации продолжения
генерации, а не только точности извлечения фактов.

На `multifieldqa_en` один из 80 промптов усечён.
Отдельная проверка без него будет дополнительной sensitivity
analysis; исходную серию задним числом не менять.

## После серии

Если четыре group sizes дадут устойчивую кривую, можно переходить
к более глубокому исследованию: отдельной чувствительности Keys
и Values, ошибок по слоям либо новой model family.
