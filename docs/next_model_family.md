# Следующая архитектура: Llama 3.2 1B Instruct

## Почему именно она

На Qwen3-1.7B эффект размера группы HQQ INT4 уже проверен
на трёх поднаборах LongBench и двух непересекающихся выборках.
Теперь нужен первый технический тест модели **другого семейства**,
а не повторение Qwen в другом размере.

Предварительный кандидат — `meta-llama/Llama-3.2-1B-Instruct`.
Это компактная instruction-tuned модель Llama с длинным заявленным
контекстом. **Но это не Qwen3-1.7B с заменённой архитектурой:**
различаются размер (1B vs 1.7B), обучение, токенизация и другие
параметры. По сравнению нельзя изолировать причинное влияние
только архитектуры. Сейчас нужна лишь проверка, запускается ли
stock HQQ QuantizedCache на Llama в нашем окружении.

Страница модели и правила доступа:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Meta требует принятия лицензии для скачивания весов на
Hugging Face. Это отдельное действие владельца HF-аккаунта.
Нельзя сохранять токены или веса в публичном GitHub.

## Команды на login-узле ВШЭ

### 1. На сайте Hugging Face

Открой страницу модели, войди в свой аккаунт, изучи и,
если согласен, прими условия доступа. Дождись подтверждения
разрешения скачать модель.

### 2. Проверить CLI Hugging Face

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
source cluster.env.example
python -m unittest discover -s tests -v
command -v hf
```

Если `hf` есть, авторизуйся в своём аккаунте командой
`hf auth login`, следуя интерактивным инструкциям. Не вставляй
токен в команды с `echo`, аргументы Slurm, логи и Git.

Если `hf` отсутствует, остановись и пришли результат
`command -v hf`: подберём способ входа для установленной версии
`huggingface_hub`, не изменяя старое рабочее окружение без нужды.

### 3. Скачать модель в workspace

```bash
export OTHER_MODEL_PATH="$(ws_find kv-cache)/models/Llama-3.2-1B-Instruct"
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir "$OTHER_MODEL_PATH"
ls "$OTHER_MODEL_PATH/config.json"
```

Не устанавливай новую версию Transformers только ради Llama:
сначала проверяем исходную 5.17.0 и stock HQQ 0.2.8.post1.

### 4. Подготовить те же исходные LongBench-задания

```bash
python -m src.prepare_family_model \
  --model "$OTHER_MODEL_PATH" \
  --name llama32-1b-instruct
wc -l data/prepared/families/llama32-1b-instruct/*.jsonl
```

Ожидаем три файла по **80** заданий (те же source_index,
что у Qwen3 held-out), но с **новыми token IDs Llama**.
Код дополнительно проверяет, что данные и commit оригинального
LongBench совпадают с предыдущей серией.

Подготовка не должна менять ни один файл Qwen3.
Если `data/prepared/families/llama32-1b-instruct` уже существует,
повторно не удаляй его: сначала проверь manifest.

### 5. Совместимость: только по три retrieval-задания

```bash
mkdir -p logs results/families/llama32-1b-instruct/smoke
sbatch --export=ALL,OTHER_MODEL_PATH="$OTHER_MODEL_PATH" \
  slurm/llama_family_smoke.sbatch
```

Здесь Slurm array `0,1,2` проверит:
- DynamicCache;
- HQQ INT4 g16;
- HQQ INT4 g64.

Каждое задание генерирует ответ только для **первых трёх**
source_index нового набора passage_retrieval_en.
Это **не** оценка качества модели, только smoke на
совместимость API, CUDA и HQQ.

Проверка после завершения:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
wc -l results/families/llama32-1b-instruct/smoke/*.jsonl
```

Только после успешных трёх `COMPLETED 0:0` можно проверять
формат ответов и решать, пригодна ли модель для полной матрицы.

## Ограничения будущего сравнения

1. Одинаковый **source_index и текст задачи**, но token IDs,
   число токенов и усечения у Qwen3 и Llama будут различаться.
2. Внутри Llama Dynamic/g16/g64 **получают одинаковые token IDs**.
3. Выбранная Llama 3.2 1B — instruction модель; тем не менее
   текущий протокол использует **raw LongBench prompt без chat template**,
   как в эксперименте Qwen3. Это осознанное предварительное
   сравнение, но если baseline плохо отвечает, сначала нужен
   отдельный эксперимент с chat template, а не произвольная
   замена формата внутри одной серии.
4. Сравнивать следует изменения качества Dynamic→INT4 **внутри
   каждой модели**, а не сами абсолютные баллы Qwen3 против Llama.
5. Первый smoke не подтверждает наличие эффекта `q_group_size`.
   Для научного сравнения нужен достаточно хороший Dynamic baseline
   и расширенная выборка.

## Какие модели можно рассмотреть позже

- SmolLM3 3B — открытая многоязычная модель с long context,
  но отдельной схемой attention; для сравнения потребуется
  проверить пути кэширования и режимы thinking.
- OLMo 2 1B — открытая модель другого семейства, однако
  стандартный context window 4096 токенов, поэтому она
  не подходит для прямого повторения нашей LongBench 16K серии.

Ссылки:
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/allenai/OLMo-2-0425-1B
