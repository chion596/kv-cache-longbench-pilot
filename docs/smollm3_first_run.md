# Смежная архитектура без одобрения Meta: SmolLM3-3B

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
