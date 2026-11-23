# Первый запуск на кластере ВШЭ

Я выполняю команды **строго по порядку**. Терминал Mac и login-узел кластера — разные среды.

## 1. На Mac я проверяю GitHub

Сначала я открываю терминал Mac:

```bash
cd ~/Downloads/kv-cache-longbench-pilot
git pull --ff-only origin main
```

После изменений на GitHub моя локальная копия может отставать — это нормально.

## 2. Подключиться к кластеру

На Mac я использую ранее работавшее SSH-подключение к кластеру ВШЭ — **ровно ту SSH-команду**, которая сработала в предыдущем исследовании. Новые ключи генерировать не нужно.

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

## 3. Я проверяю существующие веса и окружение

Я переиспользую **окружение и веса предыдущего проекта**, не меняя сам старый репозиторий.

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

Если `ws_find kv-cache` или `ls` не сработают, я остановлюсь и сохраню текст ошибки. Модель заново не скачиваю.

Если обычный `import transformers` выполняется очень долго и я прерываю его через Ctrl+C, это не доказывает повреждение библиотеки. Тогда я проверяю установленную версию без полного импорта:

```bash
python -c "from importlib.metadata import version; print('torch', version('torch')); print('transformers', version('transformers')); print('hqq', version('hqq'))"
```

Полный импорт потребуется мне позже при токенизации. Если явной ошибки нет, я дожидаюсь его завершения.

## 4. Подготовить LongBench без GPU

На login-узле, внутри `~/kv-cache-longbench-pilot`:

```bash
python -m pip install -r requirements-extra.txt
mkdir -p external
git clone https://github.com/THUDM/LongBench.git external/LongBench
git -C external/LongBench rev-parse HEAD
python -m src.download_data
```

Если `external/LongBench` уже существует, я пропускаю `git clone` и проверяю его текущий commit:

```bash
git -C external/LongBench rev-parse HEAD
```

Если сеть кластера заблокирует скачивание, я сохраню ошибку; тогда отдельно организую перенос архива с Mac.

## 5. Я готовлю маленькую выборку и проверяю тесты

```bash
source cluster.env.example
python -m src.prepare_data --model "$MODEL_PATH" --n 5 --max-input-tokens 16384
python -m unittest discover -s tests -v
mkdir -p logs results/smoke results/raw results/summary
```

После подготовки я проверяю, что существуют:

```bash
ls data/prepared/
```

Ожидаются `passage_retrieval_en.jsonl`, `multifieldqa_en.jsonl`, `multifieldqa_zh.jsonl`, `manifest.json`.

## 6. Первый GPU smoke test

**На login-узле** я отправляю работу в Slurm:

```bash
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/smoke.sbatch
```

Номер задания из ответа `Submitted batch job ...` я сохраняю.

Очередь я проверяю так:

```bash
squeue -u "$USER"
```

После завершения (замени `JOBID` на номер):

```bash
cat logs/lb-smoke-JOBID.out
cat logs/lb-smoke-JOBID.err
```

Я проверяю три ответа модели, отсутствие CUDA/HQQ ошибок, формат ответов и время. **Smoke test на трёх заданиях не является научным результатом.**

### Я проверяю ответы официальной метрикой

На login-узле, в каталоге репозитория, после успешного smoke:

```bash
python -m src.check_smoke results/smoke/*.jsonl
```

Для каждой задачи скрипт показывает эталонный ответ, сгенерированный текст, извлечённые номера абзацев и балл по оригинальной метрике LongBench. По трём примерам я не делаю выводов; при некорректном формате сначала разбираю проблему.

## 7. Полный пилот — только после разбора smoke

После того как я разберу вывод smoke:

```bash
python -m src.prepare_data --model "$MODEL_PATH" --n 40 --max-input-tokens 16384
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH" slurm/pilot.sbatch
```

Slurm array содержит девять ячеек: три датасета × три конфигурации cache.

**Этот этап я запускаю только после разбора smoke.** Если обнаружится проблема с format/prompt/length, сначала исправлю её и лишь затем зафиксирую экспериментальную настройку.

## 8. Анализ результатов

Только после успешного завершения всех девяти ячеек:

```bash
python -m src.score --longbench-code external/LongBench/LongBench
python -m src.plot
cat results/summary/dataset_scores.csv
cat results/summary/paired_deltas.csv
```

## Что я сохраняю после первого этапа

Мне нужны четыре фрагмента вывода с кластера:
- версии `torch` и `transformers`;
- результат `ls "$MODEL_PATH/config.json"`;
- результат `python -m unittest discover -s tests -v`;
- текст `logs/lb-smoke-JOBID.out` и `logs/lb-smoke-JOBID.err`.

Я не сохраняю в репозитории пароли, токены, приватные ключи и полные пользовательские конфиги.
