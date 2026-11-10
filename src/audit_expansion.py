"""CPU-аудит неиспользованных документов LongBench; не запускает inference.

Исключает опубликованные n40/n80, prepared JSONL и сохранённые результаты.
Выборки — предварительные: по одному вопросу на новый exact context.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile

from src.download_data import sha256_file

DEFAULT_DATASETS = ("passage_retrieval_en", "multifieldqa_en", "multifieldqa_zh",
                    "passage_retrieval_zh", "qasper", "hotpotqa")


class Catalog:
    def __init__(self, archive):
        self.archive = archive
        self.members = defaultdict(list)
        self.cache = {}
        for name in archive.namelist():
            if name.endswith('.jsonl'):
                self.members[Path(name).stem].append(name)

    def get(self, dataset):
        if dataset not in self.cache:
            matches = self.members.get(dataset, [])
            if len(matches) != 1:
                raise ValueError(f"Нужен один JSONL набора {dataset}: {matches}")
            raw = self.archive.read(matches[0])
            records = []
            # LF обязателен: U+0085 внутри JSON-строки не является концом записи.
            for index, line in enumerate(line for line in raw.decode('utf-8').split('\n') if line.strip()):
                row = json.loads(line)
                context = row['context']
                if not isinstance(context, str) or not context:
                    raise ValueError(f"Пустой context: {dataset}:{index}")
                records.append({'source_index': index,
                    'context_sha256': hashlib.sha256(context.encode('utf-8')).hexdigest()})
            self.cache[dataset] = (records, hashlib.sha256(raw).hexdigest())
        return self.cache[dataset]


def eligible_preview(records, used_indices, used_contexts, n, seed):
    groups = defaultdict(list)
    for row in records:
        if row['source_index'] not in used_indices and row['context_sha256'] not in used_contexts:
            groups[row['context_sha256']].append(row['source_index'])
    rng = random.Random(seed)
    selected_contexts = rng.sample(sorted(groups), min(n, len(groups)))
    selected = sorted(rng.choice(sorted(groups[context])) for context in selected_contexts)
    return {'unused_question_indices': len(records) - len(used_indices),
        'eligible_questions_on_new_contexts': sum(len(v) for v in groups.values()),
        'eligible_contexts': len(groups), 'preview_n': len(selected),
        'preview_source_indices': selected}


def audit(catalog, published, history_roots, datasets, n=80, seed=2031):
    used = defaultdict(set)
    contexts = set()
    errors = []
    files = set()
    recognized = 0
    for item in published['datasets']:
        dataset = item['dataset']
        records, member_hash = catalog.get(dataset)
        if member_hash != item['member_sha256']:
            raise ValueError(f"Архив отличается от опубликованного аудита: {dataset}")
        for index in set(item['pilot_indices']) | set(item['heldout_indices']):
            if type(index) is not int or not 0 <= index < len(records):
                raise ValueError(f"Недопустимый индекс опубликованной истории: {dataset}/{index}")
            used[dataset].add(index)
            contexts.add(records[index]['context_sha256'])
    manifests = set()
    for root in history_roots:
        root = Path(root)
        if root.exists():
            files.update(path.resolve() for path in root.rglob('*.jsonl'))
            manifests.update(path.resolve() for path in root.rglob('manifest.json'))
    for path in sorted(manifests):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            digest = meta.get('archive_sha256')
            if digest and digest != published['archive_sha256_from_upstream_lfs']:
                errors.append(f"Другой archive_sha256 в {path}")
        except (OSError, ValueError, AttributeError) as error:
            errors.append(f"Manifest {path}: {error}")
    for path in sorted(files):
        try:
            with path.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Строка {line_number} не является объектом")
                    dataset = row.get('dataset')
                    task_id = str(row.get('task_id', ''))
                    prefix, separator, suffix = task_id.rpartition(':')
                    if dataset is None and separator:
                        dataset = prefix
                    provided_hash = row.get('context_sha256')
                    if provided_hash is not None and (not isinstance(provided_hash, str) or not re.fullmatch('[0-9a-f]{64}', provided_hash)):
                        raise ValueError(f"Некорректный context hash, строка {line_number}")
                    if provided_hash:
                        contexts.add(provided_hash)
                    if dataset not in catalog.members:
                        continue
                    records, _ = catalog.get(dataset)
                    if separator and prefix != dataset:
                        raise ValueError(f"task_id не совпал с dataset, строка {line_number}")
                    index = row.get('source_index')
                    if index is None and separator and suffix.isdecimal():
                        index = int(suffix)
                    if index is None and provided_hash:
                        if provided_hash not in {r['context_sha256'] for r in records}:
                            raise ValueError(f"Context hash отсутствует в наборе, строка {line_number}")
                        recognized += 1
                        continue
                    if type(index) is not int or not 0 <= index < len(records):
                        raise ValueError(f"Недопустимый source_index, строка {line_number}")
                    actual_hash = records[index]['context_sha256']
                    if provided_hash and provided_hash != actual_hash:
                        raise ValueError(f"Context hash не совпал с архивом, строка {line_number}")
                    used[dataset].add(index)
                    contexts.add(actual_hash)
                    recognized += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"JSONL {path}: {error}")
    summaries = []
    for dataset in datasets:
        records, member_hash = catalog.get(dataset)
        preview = eligible_preview(records, used[dataset], contexts, n, seed)
        summaries.append({'dataset': dataset, 'total_questions': len(records),
            'total_contexts': len({r['context_sha256'] for r in records}),
            'excluded_question_indices': len(used[dataset]), 'member_sha256': member_hash, **preview})
    return {'history_jsonl_files': len(files), 'recognized_history_rows': recognized,
        'history_scan_errors': errors,
        'history_coverage': 'published_and_local' if files and recognized else 'published_only',
        'local_history_scan_passed': bool(files and recognized and not errors),
        'preview_only': True, 'target_documents_per_dataset': n, 'selection_seed': seed,
        'selection': 'uniform exact-context sampling, then one random question per context',
        'datasets': summaries,
        'limitations': ['Exact UTF-8 context hashes do not detect semantic duplicates or translations.',
            'Prepared but unevaluated inputs are conservatively treated as previously used.',
            'Runs missing from supplied history roots and published audit cannot be detected.',
            'Candidate previews are separate; final experiment allocation and cross-dataset deduplication remain to be fixed.']}


def save_report(out, report):
    path = Path(out)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f"Аудит уже существует с другими условиями: {path}; задайте новый --output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/raw/longbench-data.zip')
    parser.add_argument('--published-audit', default='docs/data_independence_audit_2026-11-03.json')
    parser.add_argument('--history-roots', nargs='*', default=['data/prepared', 'results'])
    parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS))
    parser.add_argument('--n', type=int, default=80)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--output', default='results/expansion_audit_2026-11-11.json')
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n должен быть положительным')
    published = json.loads(Path(args.published_audit).read_text(encoding='utf-8'))
    archive_hash = sha256_file(Path(args.archive))
    if archive_hash != published['archive_sha256_from_upstream_lfs']:
        raise ValueError('SHA-256 архива отличается от зафиксированного источника; нужен отдельный аудит версии')
    with zipfile.ZipFile(args.archive) as archive:
        report = audit(Catalog(archive), published, args.history_roots, args.datasets, args.n, args.seed)
    report.update(archive_sha256=archive_hash, source_revision=published['revision'],
                  published_audit_sha256=sha256_file(Path(args.published_audit)),
                  audit_source_sha256=sha256_file(Path(__file__)))
    save_report(args.output, report)
    for row in report['datasets']:
        print({k: v for k, v in row.items() if k not in ('member_sha256', 'preview_source_indices')})
    print('История:', report['history_coverage'], '; JSONL:', report['history_jsonl_files'],
          '; прочитано записей:', report['recognized_history_rows'], '; ошибок:', len(report['history_scan_errors']))
    print('Сохранено:', args.output)
    print('Только предварительный план; GPU не запускается. Exact-context проверка не исключает смысловые дубли.')
    if report['history_scan_errors']:
        for error in report['history_scan_errors'][:3]:
            print(error)
        parser.exit(2, 'История прочитана не полностью; итоговую независимую выборку пока не фиксируем.\n')
    if args.history_roots and not report['local_history_scan_passed']:
        parser.exit(2, 'Не найдена локальная история задач; проверьте корень репозитория и --history-roots.\n')


if __name__ == '__main__':
    main()
