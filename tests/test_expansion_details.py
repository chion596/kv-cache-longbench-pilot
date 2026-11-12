import copy
import unittest

from src.analyze_expansion import analyze_details


class ExpansionDetailsTests(unittest.TestCase):
    def fixture(self):
        config = {'models': ['a', 'b'], 'datasets': ['passage_retrieval_en', 'hotpotqa'],
                  'protocols': ['single_prefill', 'context_then_question'], 'caches': ['dynamic', 'int4_g64']}
        matrix = {}
        for model in config['models']:
            for dataset in config['datasets']:
                for protocol in config['protocols']:
                    for cache in config['caches']:
                        pred = 'Paragraph 1' if dataset == 'passage_retrieval_en' else 'yes'
                        matrix[(model, dataset, protocol, cache)] = {f'{dataset}:{i}': {'pred': pred,
                            'answers': [pred], 'context_sha256': str(i), 'middle_truncated': model == 'b' and i == 1}
                            for i in range(2)}
        metrics = {d: lambda p, a, **_: float(p == a) for d in config['datasets']}
        return matrix, config, metrics

    def test_common_subset_and_format_are_distinct_from_official_score(self):
        matrix, config, metrics = self.fixture()
        matrix[('a', 'hotpotqa', 'single_prefill', 'dynamic')]['hotpotqa:0']['pred'] = 'yes\nextra words'
        original = copy.deepcopy(matrix)
        subset, formats, _, quality = analyze_details(matrix, config, metrics)
        row = next(r for r in formats if r['model'] == 'a' and r['protocol'] == 'single_prefill' and r['cache'] == 'dynamic')
        self.assertEqual(row['official_f1_pct'], 50)
        self.assertEqual(row['first_line_f1_pct'], 100)
        self.assertEqual(row['delta_pp'], 50)
        self.assertTrue(all(r['n'] == 1 for r in subset))
        q = next(r for r in quality if r['model'] == 'a' and r['protocol'] == 'single_prefill')
        self.assertEqual(q['all_models_dynamic_full_f1_count'], 1)
        self.assertEqual(q['common_untruncated_f1_pct'], 0)
        self.assertEqual(matrix, original)

    def test_first_number_is_not_official_retrieval_metric(self):
        matrix, config, metrics = self.fixture()
        matrix[('a', 'passage_retrieval_en', 'context_then_question', 'int4_g64')]['passage_retrieval_en:0']['pred'] = 'Paragraph 1. Or 2?'
        _, _, retrieval, _ = analyze_details(matrix, config, metrics)
        row = next(r for r in retrieval if r['model'] == 'a' and r['protocol'] == 'context_then_question' and r['cache'] == 'int4_g64')
        self.assertEqual(row['official_score_pct'], 50)
        self.assertEqual(row['first_number_correct'], 2)
        self.assertEqual(row['delta_pp'], 0)


if __name__ == '__main__':
    unittest.main()
