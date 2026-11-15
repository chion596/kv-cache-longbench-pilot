import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.analyze_kv_probe import analyze_details, load_native_references
from src.common import write_jsonl
from src.download_data import sha256_file
from src.kv_probe import load_config


class ProbeDetailsTests(unittest.TestCase):
    def fixture(self):
        config = {**load_config(), 'models': ['a', 'b']}
        matrix, native = {}, {}
        for model in config['models']:
            for policy in config['policies']:
                rows = {}
                for i in range(3):
                    base = float(model == 'a' and i == 2)
                    effect = 0 if policy == 'dynamic' else 0.1 if policy.startswith('v_') else [1., 1., 9.][i]
                    if policy.startswith('kv_'):
                        effect += .2
                    rows[str(i)] = {'context_sha256': str(i), 'gold_number': i+1, 'target_ids': [1, 2],
                        'input_ids_sha256': str(i), 'gold_nll_mean': base+effect,
                        'gold_nll_per_token': [base+effect]*2, 'first_number_token_margin': -effect,
                        'first_number_token_top1': 9 if model == 'a' and i == 2 else 1, 'query_first_token': 5}
                matrix[model, policy] = rows
                if policy in ('dynamic', 'kv_g32', 'kv_g64'):
                    native[model, policy] = {str(i): {'pred': f'Paragraph {i+1}', 'output_ids': [5]} for i in range(3)}
        return matrix, config, native

    def test_direct_paired_effect_and_loo_sensitivity_are_not_confused(self):
        matrix, config, native = self.fixture()
        before = copy.deepcopy((matrix, native))
        spread, _, contrasts, common = analyze_details(matrix, config, native)
        row = next(r for r in spread if r['model'] == 'a' and r['policy'] == 'k_g64')
        self.assertAlmostEqual(row['delta_nll'], 11/3)
        self.assertEqual(row['leave_one_out_min_delta'], 1)
        self.assertEqual(row['leave_one_out_max_delta'], 5)
        self.assertAlmostEqual(row['largest_absolute_delta_share'], 9/11)
        self.assertEqual(common, 2)
        direct = next(r for r in contrasts if r['model'] == 'a' and r['metric'] == 'gold_nll_mean'
                      and r['comparison'] == 'kv_g64-k_g64')
        self.assertAlmostEqual(direct['delta'], .2)
        self.assertAlmostEqual(direct['cluster_ci95_low'], .2)
        self.assertAlmostEqual(direct['cluster_ci95_high'], .2)
        conditional = [r for r in contrasts if r['scope'] == 'common_dynamic_first_token_correct']
        self.assertTrue(all(r['n'] == 2 for r in conditional))
        self.assertEqual(len(contrasts), 32)
        self.assertEqual((matrix, native), before)

    def test_native_number_first_token_and_all_gold_tokens_are_separate(self):
        matrix, config, native = self.fixture()
        rows = matrix['a', 'kv_g32']
        rows['0']['gold_nll_per_token'] = [.1, .1]
        rows['1']['gold_nll_per_token'] = [.1, 2.]
        native['a', 'kv_g32']['0']['pred'] = 'Paragraph 99'
        native['a', 'kv_g32']['1']['pred'] = 'Paragraph 99'
        _, bridge, _, _ = analyze_details(matrix, config, native)
        row = next(r for r in bridge if r['model'] == 'a' and r['policy'] == 'kv_g32')
        self.assertEqual(row['native_first_number_correct'], 1)
        self.assertEqual(row['forced_first_number_token_top1_correct'], 2)
        self.assertEqual(row['all_gold_number_tokens_p_above_half_count'], 1)
        self.assertEqual(row['native_wrong_and_all_gold_tokens_p_above_half'], 1)

    def test_native_loader_rejects_changed_source_and_mismatched_inputs(self):
        matrix, config, _ = self.fixture()
        config['models'] = ['a']
        tasks = [{'task_id': str(i), 'input_ids_sha256': str(i), 'context_sha256': str(i)} for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'native.jsonl'
            records = [{**tasks[i], 'cache': cache, 'pred': 'Paragraph 1', 'output_ids': [5]}
                       for cache in ('dynamic', 'int4_g32', 'int4_g64') for i in range(3)]
            write_jsonl(path, records)
            manifest = {'reference_results_sha256': {str(path): sha256_file(path)}}
            with patch('src.analyze_kv_probe.load_prepared', return_value=(tasks, manifest, None)):
                self.assertEqual(len(load_native_references(matrix, config)), 3)
                records[0]['context_sha256'] = 'wrong'
                write_jsonl(path, records)
                with self.assertRaisesRegex(ValueError, 'Изменён исходный'):
                    load_native_references(matrix, config)
                manifest['reference_results_sha256'][str(path)] = sha256_file(path)
                with self.assertRaisesRegex(ValueError, 'несопоставимые'):
                    load_native_references(matrix, config)


if __name__ == '__main__':
    unittest.main()
