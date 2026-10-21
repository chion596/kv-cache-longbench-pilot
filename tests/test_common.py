import unittest
from src.common import deterministic_indices, digest_ids, middle_truncate
from src.prepare_data import build_prompt
from src.score import bootstrap_paired_ci, score_prediction


class CommonTests(unittest.TestCase):
    def test_middle_truncate(self):
        ids, truncated = middle_truncate(list(range(20)), 7)
        self.assertEqual(ids, [0, 1, 2, 16, 17, 18, 19])
        self.assertTrue(truncated)
        self.assertEqual(middle_truncate([1, 2, 3], 10), ([1, 2, 3], False))

    def test_sampling(self):
        self.assertEqual(deterministic_indices(50, 7, 2026), deterministic_indices(50, 7, 2026))
        self.assertEqual(len(deterministic_indices(50, 7, 2026)), 7)
        self.assertEqual(len(set(deterministic_indices(50, 7, 2026))), 7)

    def test_hash(self):
        self.assertEqual(digest_ids([1,2,3]), digest_ids([1,2,3]))
        self.assertNotEqual(digest_ids([1,2,3]), digest_ids([1,2,4]))

    def test_prompt(self):
        self.assertEqual(build_prompt("Context: {context} / Q: {input}",
                                      {"context": "{keep}", "input": "Who?"}),
                         "Context: {keep} / Q: Who?")

    def test_official_max_reference_scoring(self):
        row={"pred":"b", "answers":["a","b"], "all_classes":None}
        self.assertEqual(score_prediction(row, lambda a,b,**kwargs: float(a==b)), 1.)

    def test_bootstrap_equal_pairs(self):
        self.assertEqual(bootstrap_paired_ci([(1.,1.),(0.5,0.5)], iters=100), (0.,0.))


if __name__ == "__main__":
    unittest.main()
