# Первый запуск на кластере ВШЭ

Команды выполняются **строго по порядку**. Терминал Mac и login-узел кластера — разные среды.

## 1. Mac: проверить GitHub

Открой терминал Mac:

```bash
cd ~/Downloads/kv-cache-longbench-pilot
git pull --ff-only origin main
```

После изменений через GitHub локальная копия может отставать — это нормально.

## 2. Подключиться к кластеру

На Mac открой ранее работавшее SSH-подключение к кластеру ВШЭ. Используй **ровно ту SSH-команду**, с которой ты работал в предыдущем исследовании; не нужно генерировать новые ключи.

Дальнейшие команды выполняются **на login-узле кластера**, не на Mac.

```bash
cd ~
git clone https://github.com/chion596/kv-cache-longbench-pilot.git
cd ~/kv-cache-longbench-pilot
git log -1 --oneline
```

Если каталог уже существует, не клонируй повторно:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
```

## 3. Проверить уже существующие веса и окружение

Мы переиспользуем **окружение и веса предыдущего проекта**, но не изменяем сам старый репозиторий.

```bash
module load python
conda activate kvcache
python --version
python -c "import torch,transformers,hqq; print('torch:',torch.__version__,'transformers:',transformers.__version__)"
ws_find kv-cache
source cluster.env.example
echo "$MODEL_PATH"
ls "$MODEL_PATH/config.json"
```

**Остановись и пришли ошибку**, если `ws_find kv-cache` или `ls` не сработали. Не скачивай модель заново.

Если обычный `import transformers` выполняется очень долго и ты прервал его через Ctrl+C, это не доказывает повреждение библиотеки. Проверь установленную версию без полного импорта:

```bash
python -c "from importlib.metadata import version; print('torch', version('torch')); print('transformers', version('transformers')); print('hqq', version('hqq'))"
```

Полный импорт потребуется позже при токенизации, и его нужно дождаться, если не появляется явная ошибка.

## 4. Подготовить LongBench без GPU

На login-узле, внутри `~/kv-cache-longbench-pilot`:

```bash
python -m pip install -r requirements-extra.txt
mkdir -p external
git clone https://github.com/THUDM/LongBench.git external/LongBench
git -C external/LongBench rev-parse HEAD
python -m src.download_data
```

Если `external/LongBench` уже существует, пропусти `git clone` и покажи его текущий commit:

```bash
git -C external/LongBench rev-parse HEAD
```

Если скачивание заблокировано сетью кластера, пришли ошибку: отдельно организуем перенос архива с Mac.

## 5. Подготовить маленькую выборку и проверить тесты

```bash
source cluster.env.example
python -m src.prepare_data --model "$MODEL_PATH" --n 5 --max-input-tokens 16384
python -m unittest discover -s tests -v
mkdir -p logs results/smoke results/raw results/summary
```

После подготовки проверь, что существуют:

```bash
ls data/prepared/
```

Ожидаются `passage_retrieval_en.jsonl`, `multifieldqa_en.jsonl`, `multifieldqa_zh.jsonl`, `manifest.json`.

## 6. Первый GPU smoke test

**На login-узле** отправь работу в Slurm:

```bash
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/smoke.sbatch
```

Запомни номер задания из ответа `Submitted batch job ...`.

Проверить очередь:

```bash
squeue -u "$USER"
```

После завершения (замени `JOBID` на номер):

```bash
cat logs/lb-smoke-JOBID.out
cat logs/lb-smoke-JOBID.err
```

Нужно проверить три ответа модели, отсутствие CUDA/HQQ ошибок, разумный формат ответов и время. **Smoke test на трёх заданиях не является научным результатом.**

### Проверить правильность ответов официальной метрикой

На login-узле, в каталоге репозитория, после успешного smoke:

```bash
python -m src.check_smoke results/smoke/*.jsonl
```

Для каждой задачи скрипт покажет эталонный ответ, сгенерированный текст, извлечённые номера абзацев и балл по оригинальной метрике LongBench. Не делаем выводы по трём примерам; при некорректном формате ответов сначала разберём проблему.

## 7. Полный пилот — только после разбора smoke

После того как мы совместно посмотрим вывод smoke:

```bash
python -m src.prepare_data --model "$MODEL_PATH" --n 40 --max-input-tokens 16384
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/pilot.sbatch
```

Slurm array содержит девять ячеек: три датасета × три конфигурации cache.

**Не запускай этот этап сразу.** Если в smoke возникнет проблема с format/prompt/length, сначала исправим её и только затем зафиксируем экспериментальную настройку.

## 8. Анализ результатов

Только после успешного завершения всех девяти ячеек:

```bash
python -m src.score --longbench-code external/LongBench/LongBench
python -m src.plot
cat results/summary/dataset_scores.csv
cat results/summary/paired_deltas.csv
```

## Что прислать после первого этапа

Нужно всего четыре фрагмента вывода с кластера:
- версии `torch` и `transformers`;
- результат `ls "$MODEL_PATH/config.json"`;
- результат `python -m unittest discover -s tests -v`;
- текст `logs/lb-smoke-JOBID.out` и `logs/lb-smoke-JOBID.err`.

Не присылай пароли, токены, приватные ключи или полные пользовательские конфиги.
