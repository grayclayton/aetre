"""Unit and integration tests for the Road A Pre-Commit Gatekeeper hook.

Validates:
1. Clean Python files pass Tier 0 & Tier 1 with exit code 0.
2. Syntax errors are immediately rejected at Tier 0 ($0.0000) with line/col diagnostics.
3. Bytecode/invariant compilation defects are caught at Tier 1.
4. Mixed batches correctly identify defective files while passing conforming ones.
5. Structured JSON receipts conform to schema.
6. Dry-run mode returns 0 while preserving diagnostics.
7. Non-Python files and missing files are safely handled.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

# Ensure repo scripts are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scripts_dir = os.path.join(repo_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import importlib.util
hook_path = os.path.join(scripts_dir, "pre-commit-governed-gate.py")
spec = importlib.util.spec_from_file_location("pre_commit_governed_gate", hook_path)
hook_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook_mod)

check_file = hook_mod.check_file
run_precommit_gate = hook_mod.run_precommit_gate
main = hook_mod.main


class TestPreCommitGovernedGate(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_temp_file(self, filename: str, content: str) -> str:
        fpath = os.path.join(self.temp_dir.name, filename)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        return fpath

    def test_clean_file_passes(self):
        """Valid Python code passes Tier 0 & Tier 1 with exit code 0."""
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        path = self._create_temp_file("clean.py", code)

        res = check_file(path)
        self.assertTrue(res["passed"])
        self.assertEqual(res["terminal_tier"], 1)
        self.assertIsNone(res["error_type"])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([path], repo_root=self.temp_dir.name)
        self.assertEqual(ret, 0)
        self.assertIn("ALL PRE-COMMIT INVARIANTS SATISFIED", out.getvalue())

    def test_syntax_error_rejected_at_tier0(self):
        """Malformed syntax is caught at Tier 0 ($0 cost) with exact location."""
        bad_code = "def broken(\n    return 42\n"
        path = self._create_temp_file("syntax_bad.py", bad_code)

        res = check_file(path)
        self.assertFalse(res["passed"])
        self.assertEqual(res["terminal_tier"], 0)
        self.assertEqual(res["error_type"], "SyntaxError")
        self.assertIsNotNone(res["lineno"])
        self.assertIn("SyntaxError", res["diagnostic"])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([path], repo_root=self.temp_dir.name)
        self.assertEqual(ret, 1)
        output_str = out.getvalue()
        self.assertIn("DEFECTS DETECTED ON HOST", output_str)
        self.assertIn("Tier 0 (SyntaxError)", output_str)
        self.assertIn("Avoided 1 container CI execution", output_str)

    def test_compilation_error_rejected(self):
        """Null byte poisoning is rejected during compilation check."""
        fpath = os.path.join(self.temp_dir.name, "poisoned.py")
        with open(fpath, "wb") as f:
            f.write(b"x = 1\x00\ny = 2\n")

        res = check_file(fpath)
        self.assertFalse(res["passed"])
        # Either caught at Tier 0 (ast.parse rejects null byte) or Tier 1 (compile rejects)
        self.assertIn(res["terminal_tier"], (0, 1))

    def test_mixed_batch(self):
        """Batch with mixed conforming and defective files identifies the exact broken file."""
        clean1 = self._create_temp_file("clean1.py", "x = 1\n")
        clean2 = self._create_temp_file("clean2.py", "y = 2\n")
        broken = self._create_temp_file("broken.py", "def f():\n  invalid syntax :::\n")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([clean1, broken, clean2], repo_root=self.temp_dir.name)
        self.assertEqual(ret, 1)
        val = out.getvalue()
        self.assertIn("broken.py", val)
        self.assertNotIn("clean1.py: Tier", val)

    def test_json_output_mode(self):
        """JSON output mode produces valid, well-structured receipts."""
        clean = self._create_temp_file("pkg/mod.py", "val = 100\n")
        bad = self._create_temp_file("pkg/bad.py", "val = [1, 2,\n")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([clean, bad], json_mode=True, repo_root=self.temp_dir.name)
        self.assertEqual(ret, 1)

        data = json.loads(out.getvalue())
        self.assertEqual(data["status"], "FAILED")
        self.assertFalse(data["passed"])
        self.assertEqual(data["files_screened"], 2)
        self.assertEqual(data["tier_0_syntax_errors"], 1)
        self.assertEqual(data["tier_1_compilation_errors"], 0)
        self.assertEqual(data["docker_containers_avoided"], 1)
        self.assertAlmostEqual(data["estimated_savings_usd"], 0.02, places=4)
        self.assertEqual(len(data["failures"]), 1)

    def test_dry_run_mode(self):
        """Dry-run flag returns 0 even when syntax defects are discovered."""
        bad = self._create_temp_file("bad.py", "class {}:\n")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([bad], dry_run=True, repo_root=self.temp_dir.name)
        self.assertEqual(ret, 0)
        self.assertIn("HALT_AND_REJECT", out.getvalue())

    def test_non_python_files_filtered_out(self):
        """Non-Python files are skipped; returns 0 when no Python files to screen."""
        txt = self._create_temp_file("notes.txt", "Some random notes")
        json_file = self._create_temp_file("config.json", '{"key": "val"}')

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = run_precommit_gate([txt, json_file], repo_root=self.temp_dir.name)
        self.assertEqual(ret, 0)
        self.assertIn("No Python files to screen", out.getvalue())

    def test_missing_file_diagnostic(self):
        """Missing files are reported without unhandled exceptions."""
        non_existent = os.path.join(self.temp_dir.name, "missing.py")
        res = check_file(non_existent)
        self.assertFalse(res["passed"])
        self.assertEqual(res["error_type"], "FileNotFoundError")

    def test_cli_main_entrypoint(self):
        """Invoking CLI main() parses arguments correctly."""
        clean = self._create_temp_file("test_main.py", "a = 1\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ret = main([clean, "--repo-root", self.temp_dir.name])
        self.assertEqual(ret, 0)
        self.assertIn("ALL PRE-COMMIT INVARIANTS SATISFIED", out.getvalue())


if __name__ == "__main__":
    unittest.main()
