import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.audit_expansion import Catalog, audit, eligible_preview, save_report


class ExpansionAuditTests(unittest.TestCase):
    def fixture(self, root):
        path = Path(root) / 'source.zip'
        en = [{'context': s} for s in ('A', 'B\u0085text', 'B\u0085text', 'C', 'D', 'E')]
        other = [{'context': s} for s in ('A', 'F', 'G', 'G', 'H')]
        raw = ('\n'.join(json.dumps(row, ensure_ascii=False) for row in en) + '\n').encode()
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('data/multifieldqa_en.jsonl', raw)
            archive.writestr('data/hotpotqa.jsonl', '\n'.join(json.dumps(row) for row in other) + '\n')
        published = {'archive_sha256_from_upstream_lfs': 'test', 'datasets': [{
            'dataset': 'multifieldqa_en', 'member_sha256': hashlib.sha256(raw).hexdigest(),
            'pilot_indices': [0], 'heldout_indices': [1]}]}
        return path, published

    def test_question_disjointness_does_not_imply_new_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path, published = self.fixture(temp)
            with zipfile.ZipFile(path) as archive:
                report = audit(Catalog(archive), published, [], ['multifieldqa_en'], n=80)
            row = report['datasets'][0]
            self.assertEqual(row['unused_question_indices'], 4)
            self.assertEqual(row['eligible_questions_on_new_contexts'], 3)
            self.assertEqual(row['eligible_contexts'], 3)
            self.assertEqual(row['preview_source_indices'], [3, 4, 5])
            self.assertFalse(report['local_history_scan_passed'])

    def test_history_and_cross_dataset_contexts_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            path, published = self.fixture(temp)
            history = Path(temp) / 'history'
            history.mkdir()
            (history / 'model.jsonl').write_text(json.dumps({'dataset': 'multifieldqa_en',
                'source_index': 3, 'task_id': 'multifieldqa_en:3'}) + '\n')
            with zipfile.ZipFile(path) as archive:
                report = audit(Catalog(archive), published, [history], ['multifieldqa_en', 'hotpotqa'], n=80)
            self.assertTrue(report['local_history_scan_passed'])
            self.assertEqual(report['datasets'][0]['preview_source_indices'], [4, 5])
            other = report['datasets'][1]
            self.assertEqual(other['eligible_contexts'], 3)
            self.assertNotIn(0, other['preview_source_indices'])

    def test_malformed_history_does_not_produce_a_passed_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            path, published = self.fixture(temp)
            history = Path(temp) / 'history'
            history.mkdir()
            (history / 'partial.jsonl').write_text('{"dataset":"multifieldqa_en","source_index":3}\n{')
            with zipfile.ZipFile(path) as archive:
                report = audit(Catalog(archive), published, [history], ['multifieldqa_en'])
            self.assertFalse(report['local_history_scan_passed'])
            self.assertEqual(len(report['history_scan_errors']), 1)

    def test_changed_archive_or_context_hash_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, published = self.fixture(temp)
            history = Path(temp) / 'history'
            history.mkdir()
            (history / 'wrong.jsonl').write_text(json.dumps({'dataset': 'multifieldqa_en',
                'source_index': 3, 'context_sha256': '0' * 64}) + '\n')
            (history / 'manifest.json').write_text('{"archive_sha256":"different"}')
            with zipfile.ZipFile(path) as archive:
                report = audit(Catalog(archive), published, [history], ['multifieldqa_en'])
                published['datasets'][0]['member_sha256'] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'Архив отличается'):
                    audit(Catalog(archive), published, [], ['multifieldqa_en'])
            self.assertEqual(len(report['history_scan_errors']), 2)
            self.assertFalse(report['local_history_scan_passed'])

    def test_uniform_preview_has_one_question_per_context_and_is_repeatable(self):
        rows = [{'source_index': i, 'context_sha256': str(i // 2)} for i in range(20)]
        a = eligible_preview(rows, set(), set(), 4, 2031)
        b = eligible_preview(rows, set(), set(), 4, 2031)
        self.assertEqual(a, b)
        selected = a['preview_source_indices']
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({rows[i]['context_sha256'] for i in selected}), 4)

    def test_report_cannot_overwrite_a_different_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            save_report(path, {'n': 1})
            save_report(path, {'n': 1})
            with self.assertRaises(FileExistsError):
                save_report(path, {'n': 2})
            self.assertEqual(json.loads(path.read_text()), {'n': 1})


if __name__ == '__main__':
    unittest.main()
