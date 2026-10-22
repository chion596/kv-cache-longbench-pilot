"""Локальные тесты сохранения одинаковых исходных заданий для других моделей."""
import unittest

from src.prepare_family_model import original_heldout_indices


class FamilyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "requested_n": 3,
            "datasets": {
                "passage_retrieval_en": {
                    "selected_source_indices": [2, 7, 9],
                    "excluded_source_indices": [0, 4, 6]
                }
            },
        }

    def test_preserves_same_source_indices(self):
        self.assertEqual(
            original_heldout_indices(self.manifest, "passage_retrieval_en", 20),
            [2, 7, 9],
        )

    def test_rejects_overlap(self):
        self.manifest["datasets"]["passage_retrieval_en"]["excluded_source_indices"].append(7)
        with self.assertRaisesRegex(ValueError, "пересечение"):
            original_heldout_indices(self.manifest, "passage_retrieval_en", 20)

    def test_rejects_duplicate_source_indices(self):
        self.manifest["datasets"]["passage_retrieval_en"]["selected_source_indices"] = [2, 2, 9]
        with self.assertRaisesRegex(ValueError, "повторные"):
            original_heldout_indices(self.manifest, "passage_retrieval_en", 20)

    def test_rejects_bad_index(self):
        self.manifest["datasets"]["passage_retrieval_en"]["selected_source_indices"] = [2, 7, 21]
        with self.assertRaisesRegex(ValueError, "неверные индексы"):
            original_heldout_indices(self.manifest, "passage_retrieval_en", 20)


if __name__ == "__main__":
    unittest.main()
