"""CPU-подготовка первых 20 фиксированных retrieval-входов expansion_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import write_jsonl
from src.download_data import sha256_file
from src.expansion import load_config as expansion_config
from src.kv_probe import PREPARED, cell, load_config, load_reference, prepare_task
from src.protocol import digest_json
from src.run_expansion import load_inputs
from src.run_protocol import output_lock


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=config['models'], required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    tasks, old, old_path, selection, _ = load_inputs(cell(args.name, 'dynamic', config), args.model, expansion_config())
    # Порядок берём из зафиксированной selection, а не из результатов/качества.
    indices = selection['datasets'][config['dataset']]['source_indices'][:config['n']]
    by_id = {t['source_index']: t for t in tasks}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tasks = [prepare_task(by_id[i], tokenizer, config) for i in indices]
    refs, ref_hashes = load_reference(args.name, tasks, old, old_path, selection)
    folder = PREPARED/args.name
    with output_lock(folder/'manifest.json'):
        path = folder/'tasks.jsonl'
        expected = {'series': config['series'], 'config_sha256': digest_json(config), 'model_name': args.name,
            'model_artifacts': old['model_artifacts'], 'versions': old['versions'], 'hidden_layers': old['hidden_layers'],
            'reference_manifest_path': str(old_path), 'reference_manifest_sha256': sha256_file(old_path),
            'reference_tasks_path': str(old_path.parent/f"{config['dataset']}.jsonl"),
            'reference_tasks_sha256': old['datasets'][config['dataset']]['file_sha256'],
            'reference_selection_path': 'data/prepared/expansion_v1/selection.json',
            'reference_selection_sha256': old['selection_sha256'],
            'reference_results_sha256': ref_hashes, 'reference_first_tokens': refs,
            'task_ids': [t['task_id'] for t in tasks],
            'task_hashes': {t['task_id']: digest_json(t) for t in tasks},
            'prepare_source_sha256': sha256_file(Path(__file__))}
        manifest_path = folder/'manifest.json'
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if not path.exists() or saved != {**expected, 'tasks_sha256': sha256_file(path)}:
                raise ValueError('Условия подготовки изменились; входы не перезаписываем')
            from src.common import read_jsonl
            if list(read_jsonl(path)) != tasks:
                raise ValueError('Подготовленные задачи изменились')
            print('Уже подготовлено:', manifest_path)
            return
        if path.exists():
            raise FileExistsError(f'Остался tasks без manifest; проверьте вручную: {path}')
        write_jsonl(path, tasks)
        expected['tasks_sha256'] = sha256_file(path)
        with manifest_path.open('x') as handle:
            handle.write(json.dumps(expected, ensure_ascii=False, indent=2)+'\n')
    print('Подготовлено:', folder, 'документов=', len(tasks),
          'max(query+prefix+target)=', max(len(t['input_ids'])-t['context_split']+
              len(t['forced_prefix_ids'])+len(t['target_ids']) for t in tasks))


if __name__ == '__main__':
    main()
