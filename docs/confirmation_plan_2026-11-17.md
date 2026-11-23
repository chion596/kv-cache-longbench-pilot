**Следующий этап после mixed_precision_v1: независимые документы и ZH, 17 ноября 2026.**

**Обновление 21 ноября:** CPU-preflight и новый residual1024 протокол зафиксированы,
ZH smoke прошёл (8 ячеек / 24 технические проверки). Full на 120 документах
ещё не запускался. Актуальные выводы и команда находятся в [протоколе](zh_confirmation_protocol_2026-11-18.md)
и [отчёте smoke](zh_confirmation_smoke_2026-11-21.md). Текст ниже сохраняет
первоначальный план на дату 17 ноября.

**Обновление 18 ноября:** [CPU-preflight получен](zh_preflight_results_2026-11-18.md).
120 новых контекстов доступны, но один query+32=847 не помещается в residual512.
Подготовлена отдельная [zh_confirmation_v1 с общим residual1024](zh_confirmation_protocol_2026-11-18.md),
без исключения случаев. Ниже сохранён первоначальный план до проверки;
актуальные команды запуска — в новом протоколе.

Статус на 17 ноября: CPU-preflight реализован; выборка и GPU runner новой серии
ещё не зафиксированы. Сначала нужны актуальная история кластера и tokenizer
budgets. Завершённый [результат смешанной точности](mixed_precision_results_2026-11-17.md)
получен на development-контекстах, уже использованных при выборе гипотезы.

**Предлагаемая проверка.** 120 ещё не использованных exact context набора
passage_retrieval_zh, seed=2043, по одному вопросу на context; Qwen3-1.7B/4B;
Dynamic, K4/V4, K8/V4, K4/V8; native split prefill, group=64, до 16 384 input
токенов и 32 токена свободного ответа. Всего **960 ответов**.
K16/V4 на этом этапе не нужен для primary matched-memory сравнения.
Состав политик задан до ZH-генераций и не выбирается по их результатам.

Этим этапом я проверяю, переносится ли результат на китайские тексты.
EN и ZH содержат разные документы и длины, а не параллельные переводы;
различие нельзя приписывать одному только языку. Exact hashes не исключают
смысловые дубли, переводы и pretraining contamination.

**Почему 120.** Это расширяет число новых документов относительно прежних
n=80 и остаётся подмножеством набора из 200. Primary — K8/V4−K4/V8 при
равных фактических cache bytes; отдельно — K8/V4−Dynamic и официальный
score, EOS, первый номер, truncation. Не объявляем non-inferiority без
заранее обоснованной допустимой потери. Даже нулевой empirical-bootstrap
интервал не доказывает отсутствие редких будущих ухудшений.

**CPU-preflight.** На login-узле:

```bash
cd ~/kv-cache-longbench-pilot
git pull --ff-only
module load python
conda activate kvcache
bash tools/check_zh_confirmation.sh
```

Скрипт не загружает веса в модель, не делает inference, не отправляет Slurm:

- проверяет SHA-256 официального архива и parent model/tokenizer/version
  fingerprints относительно mixed_precision_v1;
- проверяет завершённую mixed full-матрицу и печатает число поэлементных
  потерь, улучшений, одинаковых scores и output IDs K8/V4 против Dynamic;
- заново сканирует published+local prepared/results history и исключает
  уже использованные индексы и exact context hashes, включая другие наборы;
- предлагает 120 новых контекстов без фильтрации по исходам или длинам;
- токенизирует их официальным ZH prompt в native no-thinking chat и проверяет
  запас `query + 32 < residual512` у обеих локальных Qwen;
- сохраняет immutable JSON `results/zh_confirmation_preflight_2026-11-17.json`.

Для ZH используется официальный
[retrieval_zh_score](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py),
разбирающий gold `段落N`, и отдельный
[официальный prompt](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json).
Его [предел ответа — 32](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json).
Код и шаблоны на кластере должны совпасть по hashes с parent-серией.

Я проверяю два условия: 120 доступных контекстов при полном history scan
и отсутствие overflow `query+32`. Если residual недостаточен, я не исключаю
вопрос, а фиксирую новый общий residual в отдельной серии. При неполной
истории я не фиксирую выборку.
Preview JSON ещё не является подготовленным inference manifest.

Повторный вызов с неизменными условиями идемпотентен. Если history изменилась,
я сохраняю прежний отчёт и задаю новое имя, например через
`ZH_PREFLIGHT_OUTPUT=results/zh_preflight_2.json bash tools/check_zh_confirmation.sh`.
На 17 ноября GPU-команд новой серии ещё не было; они подготовлены после
проверки доступности и budgets 18 ноября. Качество на ZH не использовано
для выбора этих кандидатов.
