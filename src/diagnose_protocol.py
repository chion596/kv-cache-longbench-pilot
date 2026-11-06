"""Показать готовность результатов и логи Slurm; вычисления не запускаются."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

from src.common import read_jsonl
from src.protocol import cells, load_config, result_path

LOG_NAME = re.compile(r"lb-protocol-v1-(\d+)_(\d+)\.(err|out)$")


def result_status(path, expected_n):
    path = Path(path)
    if not path.is_file():
        return "missing", 0, ""
    try:
        rows = list(read_jsonl(path))
        ids = [row["task_id"] for row in rows]
        if len(set(ids)) != len(ids):
            return "invalid", len(rows), "Повторные task_id"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "invalid", 0, str(error)
    if len(rows) < expected_n:
        return "partial", len(rows), ""
    if len(rows) > expected_n:
        return "invalid", len(rows), f"Ожидалось {expected_n} ответов"
    return "complete", len(rows), ""


def select_logs(logs_dir, matrix, job_id=None):
    """По одному представительному error log на модель из указанного/последнего массива."""
    found = []
    for path in Path(logs_dir).glob("lb-protocol-v1-*_*.err"):
        match = LOG_NAME.fullmatch(path.name)
        if match and (job_id is None or int(match[1]) == job_id):
            found.append((path.stat().st_mtime_ns, int(match[1]), int(match[2]), path))
    if not found:
        return job_id, []
    if job_id is None:
        job_id = max(found)[1]
    active = [item for item in found if item[1] == job_id]
    chosen = {}
    # Предпочитаем ненулевой stderr, затем наиболее свежую задачу модели.
    for item in sorted(active, key=lambda value: (value[3].stat().st_size > 0, value[0]), reverse=True):
        _, _, index, path = item
        label = matrix[index]["model_name"] if 0 <= index < len(matrix) else f"cell-{index}"
        if label not in chosen:
            chosen[label] = path
    return job_id, list(chosen.items())


def tail(path, lines):
    path = Path(path)
    if not path.is_file():
        return "Файла нет"
    # Небольшой хвост даже у большого stdout; не читаем весь лог в память.
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:]) or "Файл пуст"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--job-id", type=int, help="ID массива Slurm, не отдельной дочерней задачи")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--results-dir", help="По умолчанию results/protocol_v1/<phase>")
    args = parser.parse_args()
    config = load_config()
    matrix = cells(config)
    root = Path(args.results_dir) if args.results_dir else Path("results/protocol_v1") / args.phase
    expected_n = config["smoke_n"] if args.phase == "smoke" else config["n"]
    counts, problems = Counter(), []
    for index, cell in enumerate(matrix):
        status, n, detail = result_status(result_path(root, cell), expected_n)
        counts[status] += 1
        if status != "complete":
            problems.append((index, cell, status, n, detail))
    print("Каталог:", root.resolve())
    print(f"Файлы по числу ответов: полных={counts['complete']}/{len(matrix)}, "
          f"нет={counts['missing']}, неполных={counts['partial']}, повреждённых={counts['invalid']}")
    for index, cell, status, n, detail in problems[:6]:
        print(f"  cell={index}: {cell['model_name']}/{cell['dataset']}/{cell['protocol']}/{cell['cache']}: "
              f"{status}, ответов={n}/{expected_n} {detail}")
    job_id, logs = select_logs(args.logs_dir, matrix, args.job_id)
    if not logs:
        print("Логи массива не найдены. На кластере проверьте каталог logs и ID массива.")
    else:
        print(f"Массив Slurm: {job_id}. Проверка статусов:\n"
              f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed")
        for model, path in logs:
            print(f"\n{model}: {path} (последние 80 строк stderr)")
            print(tail(path, 80))
            stdout = path.with_suffix(".out")
            print(f"{stdout} (последние 30 строк stdout)")
            print(tail(stdout, 30))
    print("\nДиагностика только читает файлы. Сопоставимость полной матрицы проверяет src.score_protocol.")


if __name__ == "__main__":
    main()
