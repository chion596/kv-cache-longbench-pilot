"""Тесты создания независимой выборки: данные и GPU не требуются."""
import json
import tempfile
import unittest
from pathlib import Path

from src.common import deterministic_indices, write_jsonl
from src.prepare_heldout import load_original_selection, select_heldout_indices


class HeldoutSamplingTests(unittest.TestCase):
    def test_heldout_is_disjoint_and_deterministic(self):
        original = deterministic_indices(200, 40, 2026)
        selected = select_heldout_indices(200, original, 80, 2027)
        self.assertEqual(len(selected), 80)
        self.assertEqual(len(set(selected)), 80)
        self.assertFalse(set(selected) & set(original))
        self.assertEqual(
            selected, select_heldout_indices(200, original, 80, 2027)
        )

    def test_smallest_dataset_has_enough_unused_rows(self):
        original = deterministic_indices(150, 40, 2026)
        selected = select_heldout_indices(150, original, 80, 2027)
        self.assertEqual(len(selected), 80)
        self.assertTrue(set(selected).isdisjoint(original))

    def test_fails_if_new_sample_is_too_large(self):
        with self.assertRaisesRegex(ValueError, "Недостаточно"):
            select_heldout_indices(150, range(40), 111, 2027)

    def test_fails_if_original_has_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "повторные"):
            select_heldout_indices(100, [1, 1], 10, 2027)

    def test_reads_and_verifies_fixed_original_selection(self):
        dataset = "passage_retrieval_en"
        original = deterministic_indices(200, 40, 2026)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / f"{dataset}.jsonl", [
                {"dataset": dataset, "source_index": index,
                 "task_id": f"{dataset}:{index}"}
                for index in original
            ])
            found = load_original_selection(root, dataset, 200, 40, 2026)
            self.assertEqual(found, original)

            modified = [
                {"dataset": dataset, "source_index": i,
                 "task_id": f"{dataset}:{i}"}
                for i in original
            ]
            modified[-1] = {
                "dataset": dataset, "source_index": 199,
                "task_id": f"{dataset}:199"
            }
            write_jsonl(root / f"{dataset}.jsonl", modified)
            with self.assertRaisesRegex(ValueError, "не соответствуют"):
                load_original_selection(root, dataset, 200, 40, 2026)


if __name__ == "__main__":
    unittest.main()
