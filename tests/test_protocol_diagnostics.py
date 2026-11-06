import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common import write_jsonl
from src.diagnose_protocol import result_status, select_logs, tail
from src.protocol import cells, code_revision, load_config
from src.score_protocol import IncompleteMatrixError, main, read_matrix


class ProtocolDiagnosticsTests(unittest.TestCase):
    def test_code_revision_reads_checkout_without_git_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".git/refs/heads").mkdir(parents=True)
            (root / ".git/HEAD").write_text("ref: refs/heads/main\n")
            (root / ".git/refs/heads/main").write_text("a" * 40 + "\n")
            with patch("subprocess.check_output", side_effect=FileNotFoundError("git")) as command:
                self.assertEqual(code_revision(root, environ={}), "a" * 40)
            command.assert_not_called()

    def test_code_revision_supports_packed_and_detached_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".git").mkdir()
            (root / ".git/HEAD").write_text("ref: refs/heads/main\n")
            (root / ".git/packed-refs").write_text("# packed\n" + "b" * 40 + " refs/heads/main\n")
            self.assertEqual(code_revision(root, environ={}), "b" * 40)
            (root / ".git/HEAD").write_text("c" * 40 + "\n")
            self.assertEqual(code_revision(root, environ={}), "c" * 40)

    def test_code_revision_supports_worktree_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "checkout"
            root.mkdir()
            git_dir = base / "metadata/worktrees/run"
            git_dir.mkdir(parents=True)
            (root / ".git").write_text("gitdir: ../metadata/worktrees/run\n")
            (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
            (git_dir / "commondir").write_text("../..\n")
            (base / "metadata/refs/heads").mkdir(parents=True)
            (base / "metadata/refs/heads/main").write_text("d" * 40 + "\n")
            self.assertEqual(code_revision(root, environ={}), "d" * 40)

    def test_code_revision_uses_submitted_hash_or_allows_missing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(code_revision(temp, {"PROTOCOL_CODE_COMMIT": "E" * 40}), "e" * 40)
            self.assertIsNone(code_revision(temp, environ={}))
            with self.assertRaisesRegex(ValueError, "полный hash"):
                code_revision(temp, {"PROTOCOL_CODE_COMMIT": "main"})

    def test_missing_matrix_lists_count_without_claiming_results(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(IncompleteMatrixError) as caught:
                read_matrix(temp, load_config(), "smoke")
            self.assertIn("36 из 36", str(caught.exception))
            self.assertIn("src.diagnose_protocol --phase smoke", str(caught.exception))

    def test_scoring_cli_exits_with_actionable_message(self):
        stderr = io.StringIO()
        with patch("sys.argv", ["score_protocol", "--phase", "smoke"]), \
                patch("src.score_protocol.read_matrix", side_effect=IncompleteMatrixError("Матрица не готова")), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(stderr.getvalue(), "Матрица не готова\n")

    def test_partial_and_invalid_jsonl_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "answers.jsonl"
            self.assertEqual(result_status(path, 3)[:2], ("missing", 0))
            path.touch()
            self.assertEqual(result_status(path, 3)[:2], ("partial", 0))
            write_jsonl(path, [{"task_id": "one"}, {"task_id": "two"}])
            self.assertEqual(result_status(path, 3)[:2], ("partial", 2))
            write_jsonl(path, [{"task_id": "one"}] * 3)
            self.assertEqual(result_status(path, 3)[0], "invalid")
            path.write_text('{"task_id":')
            self.assertEqual(result_status(path, 3)[0], "invalid")
            path.write_text('1\n')
            self.assertEqual(result_status(path, 3)[0], "invalid")

    def test_log_selection_honors_array_id_and_covers_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for job in (100, 200):
                for cell in (0, 1, 12, 24):
                    (root / f"lb-protocol-v1-{job}_{cell}.err").write_text("failure" if cell != 1 else "")
            job, logs = select_logs(root, cells(load_config()), 100)
            self.assertEqual(job, 100)
            self.assertEqual({model for model, _ in logs}, set(load_config()["models"]))
            self.assertTrue(all("-100_" in path.name for _, path in logs))
            self.assertTrue(all(path.stat().st_size > 0 for _, path in logs))

    def test_tail_returns_recent_lines_and_does_not_modify_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "log.err"
            content = b"line\n" * 20000 + b"first\nlast\n"
            path.write_bytes(content)
            self.assertEqual(tail(path, 2), "first\nlast")
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(tail(path.with_suffix(".out"), 2), "Файла нет")


if __name__ == "__main__":
    unittest.main()
