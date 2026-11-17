import copy
import unittest

from src.check_zh_confirmation import DATASET, audit_ready, native_pair_details, tokenization_summary


class CharChatTokenizer:
    chat_template = 'enable_thinking'

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return '<user>'+messages[0]['content']+'</user><assistant>'

    def encode(self, text, **kwargs):
        return list(map(ord, text))


class ZhPreflightTests(unittest.TestCase):
    def test_tokenizer_budget_is_checked_without_dropping_long_queries(self):
        raw = [{'context': '文档内容。'*100, 'input': '摘要。', 'answers': ['段落1']},
               {'context': '另一个文档。'*100, 'input': '长'*600, 'answers': ['段落2']}]
        before = copy.deepcopy(raw)
        summary = tokenization_summary(raw, [0, 1], '{context}\n摘要：{input}\n答案：', CharChatTokenizer())
        self.assertEqual(summary['n'], 2)
        self.assertEqual(summary['query_plus_answer_ge_512_count'], 1)
        self.assertGreater(summary['minimum_residual_strict'], 512)
        self.assertEqual(raw, before)
        raw[0]['answers'] = ['Paragraph 1']
        with self.assertRaisesRegex(ValueError, 'ZH gold'):
            tokenization_summary(raw, [0], '{context} {input}', CharChatTokenizer())

    def test_equal_scores_do_not_imply_identical_output_ids(self):
        metric = lambda prediction, truth, **_: float(prediction.startswith(truth))
        rows = {str(i): {'answers': [str(i)], 'pred': str(i), 'output_ids': [i]} for i in range(2)}
        other = copy.deepcopy(rows)
        other['0'].update(pred='0 extra', output_ids=[0, 7])
        matrix = {('a', DATASET, 'dynamic'): rows, ('a', DATASET, 'k8_v4'): other}
        before = copy.deepcopy(matrix)
        out = native_pair_details(matrix, {DATASET: metric})[0]
        self.assertEqual(out['score_equal_count'], 2)
        self.assertEqual(out['output_ids_identical'], 1)
        self.assertEqual(out['score_lower_count'], 0)
        self.assertEqual(matrix, before)

    def test_preview_requires_complete_history_and_full_target_count(self):
        report = {'local_history_scan_passed': True, 'history_scan_errors': [], 'datasets': [{'preview_n': 120}]}
        self.assertTrue(audit_ready(report, 120))
        report['datasets'][0]['preview_n'] = 119
        self.assertFalse(audit_ready(report, 120))
        report['datasets'][0]['preview_n'] = 120
        report['history_scan_errors'] = ['incomplete source file']
        self.assertFalse(audit_ready(report, 120))
        report['history_scan_errors'] = []; report['local_history_scan_passed'] = False
        self.assertFalse(audit_ready(report, 120))


if __name__ == '__main__':
    unittest.main()
