"""Comprehensive regression and verification tests for the SWE-bench Gatekeeper.

Validates both:
1. Benchmark Acceptance & Contract Tests (15-PR corpus with Knapsack queue triage)
2. Low-Level Pure-Code Gate Regression Tests (raw string patch execution, ZeroDivisionError
   runtime exceptions at Tier 1, boundary divergence at Tier 2, and CLI stdout report assertions).
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

from benchmarks.run_swebench_benchmark import create_swebench_pr_corpus, run_swebench_benchmark
from governed_agent import TieredVerificationGate
from governed_agent.cli import main


class TestSWEBenchGatekeeper(unittest.TestCase):

    def setUp(self):
        self.prs = create_swebench_pr_corpus()
        self.gate = TieredVerificationGate(dry_run=True)

    def test_swebench_corpus_funnel_and_economics(self):
        """Acceptance Test: Verifies exact 15-PR screening funnel, Knapsack triage, and >99% compute savings."""
        report = self.gate.gate_batch(self.prs, capacity_K=3, review_cost_k=1)
        funnel = report.funnel
        econ = report.economics

        # 1. Funnel assertions
        self.assertEqual(funnel["total_incoming_candidates"], 15)
        self.assertEqual(funnel["tier_0_syntax_screened"], 3, "3 syntax errors must be caught at Tier 0")
        self.assertEqual(funnel["tier_1_crude_bug_screened"], 3, "3 crude bugs must be caught at Tier 1")
        self.assertEqual(funnel["tier_2_mutation_screened"], 2, "2 subtle boundary bugs must be caught at Tier 2")
        self.assertEqual(funnel["tier_3_qualified"], 7, "7 conforming PRs must reach Tier 3")
        self.assertEqual(funnel["knapsack_admitted"], 3, "Top 3 admitted under capacity K=3")
        self.assertEqual(funnel["knapsack_deprioritized"], 4, "4 lower-utility PRs buffered")

        # 2. Knapsack triage assertions
        admitted_ids = [c["candidate_id"] for c in report.admitted_candidates]
        self.assertEqual(admitted_ids, ["PR-101", "PR-102", "PR-103"])

        # First excluded candidate is PR-104 with expected utility 0.028
        self.assertAlmostEqual(econ["capacity_shadow_price_lambda"], 0.028000, places=4)
        self.assertGreater(econ["total_welfare_w"], 0.10)

        # 3. Economic & Docker evasion assertions
        self.assertEqual(econ["docker_invocations_avoided"], 8, "8 defective PRs bypassed Docker on host")
        self.assertAlmostEqual(econ["docker_invocations_avoided_pct"], 53.33, delta=0.5)
        self.assertGreater(econ["cost_savings_pct"], 99.0, "Cost savings must exceed 99%")
        self.assertLess(econ["total_governed_cost_usd"], 0.10)

    def test_swebench_cli_stdout_validation(self):
        """CLI Test: Captures stdout to validate the generated report contents, funnel, and shadow price."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            json.dump(self.prs, f)
            corpus_path = f.name

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ret = main(["gate", corpus_path, "--capacity", "3"])
            self.assertEqual(ret, 0)
            output = buf.getvalue()

            # Validate that the CLI prints the full structured report
            self.assertIn("ROAD A: TIERED VERIFICATION GATE BATCH REPORT", output)
            self.assertIn("[Tier 0 AST Syntax Screened]       3 PRs", output)
            self.assertIn("[Tier 1 Fast Invariants Screened]   3 PRs", output)
            self.assertIn("[Tier 2 Mutation Oracles Screened] 2 PRs", output)
            self.assertIn("[Tier 3 Qualified for Docker]      7 PRs", output)
            self.assertIn("[Knapsack Admitted to Review]      3 PRs", output)
            self.assertIn("[Knapsack Deprioritized]           4 PRs", output)
            self.assertIn("PR-101", output)
            self.assertIn("PR-102", output)
            self.assertIn("PR-103", output)
            self.assertIn("Review Capacity Shadow Price lambda_K:  $0.028000", output)
            self.assertIn("Cost Savings:                $29.98", output)
        finally:
            if os.path.exists(corpus_path):
                os.remove(corpus_path)

    def test_pure_code_tier1_runtime_exception_rejection(self):
        """Regression Test: Compiles and executes raw Python code on host; catches ZeroDivisionError at Tier 1."""
        cand_code = "def chunk_encode(data):\n    return len(data) // len(data)\n"
        ref_code = "def chunk_encode(data):\n    return 1 if len(data) > 0 else 0\n"
        probes = [
            {"name": "empty_str_probe", "input": {"data": ""}, "tier": 1},
        ]

        receipt = self.gate.verify_candidate(
            candidate_id="PR-REAL-CRUDE",
            candidate_code=cand_code,
            reference_code=ref_code,
            probes=probes,
        )

        self.assertFalse(receipt.passed)
        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 1, "Must fail at Tier 1 on host")
        self.assertFalse(receipt.docker_invoked, "Must NOT invoke Docker")
        self.assertTrue(receipt.short_circuited)
        self.assertIn("ZeroDivisionError", receipt.diagnostics[-1], "Diagnostic must record runtime exception from patch code")

    def test_pure_code_tier2_boundary_divergence_rejection(self):
        """Regression Test: Passes Tier 1 smoke test, then catches subtle off-by-one boundary defect at Tier 2."""
        cand_code = "def cookie_expiry(secs):\n    return secs - 1 if secs > 0 else 0\n"
        ref_code = "def cookie_expiry(secs):\n    return secs if secs > 0 else 0\n"
        probes = [
            {"name": "t1_smoke_zero", "input": {"secs": 0}, "tier": 1},
            {"name": "t2_boundary_hour", "input": {"secs": 3600}, "tier": 2},
        ]

        receipt = self.gate.verify_candidate(
            candidate_id="PR-REAL-SUBTLE",
            candidate_code=cand_code,
            reference_code=ref_code,
            probes=probes,
        )

        self.assertFalse(receipt.passed)
        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 2, "Must pass Tier 1 smoke and fail at Tier 2 boundary")
        self.assertFalse(receipt.docker_invoked)
        self.assertTrue(receipt.short_circuited)

        # Confirm Tier 1 probe passed and Tier 2 probe failed
        self.assertTrue(receipt.outcomes[1].passed, "Tier 1 smoke check should have passed")
        self.assertFalse(receipt.outcomes[2].passed, "Tier 2 boundary check should have failed")
        self.assertIn("Mismatch: 3599 != 3600", receipt.diagnostics[-1], "Diagnostic must record exact return value mismatch")

    def test_pure_code_batch_funnel_without_synthetic_flags(self):
        """Regression Test: Multi-PR batch with NO synthetic flags; validates full funnel execution."""
        batch = [
            # 1. Real Syntax Error (missing colon)
            {
                "candidate_id": "PR-RAW-SYNTAX",
                "patch_code": "def bad_fn(x)\n    return x + 1\n",
                "reference_code": "def bad_fn(x):\n    return x + 1\n",
            },
            # 2. Real Crude Bug (runtime ZeroDivisionError)
            {
                "candidate_id": "PR-RAW-CRUDE",
                "patch_code": "def div_fn(x):\n    return 10 // x\n",
                "reference_code": "def div_fn(x):\n    return 10 // (x if x != 0 else 1)\n",
                "probes": [{"name": "div_zero", "input": {"x": 0}, "tier": 1}],
            },
            # 3. Real Subtle Bug (off-by-one comparison)
            {
                "candidate_id": "PR-RAW-SUBTLE",
                "patch_code": "def cmp_fn(a, b):\n    return a >= b\n",
                "reference_code": "def cmp_fn(a, b):\n    return a > b\n",
                "probes": [
                    {"name": "smoke", "input": {"a": 5, "b": 2}, "tier": 1},
                    {"name": "boundary", "input": {"a": 3, "b": 3}, "tier": 2},
                ],
            },
            # 4. Real Conforming PR (identical logic)
            {
                "candidate_id": "PR-RAW-PASS",
                "patch_code": "def add_fn(a, b):\n    return a + b\n",
                "reference_code": "def add_fn(a, b):\n    return a + b\n",
                "probes": [
                    {"name": "t1_add", "input": {"a": 1, "b": 2}, "tier": 1},
                    {"name": "t2_add", "input": {"a": 10, "b": 20}, "tier": 2},
                    {"name": "t3_add", "input": {"a": 100, "b": 200}, "tier": 3},
                ],
                "expected_utility": 0.035,
            },
        ]

        report = self.gate.gate_batch(batch, capacity_K=2)
        funnel = report.funnel

        self.assertEqual(funnel["total_incoming_candidates"], 4)
        self.assertEqual(funnel["tier_0_syntax_screened"], 1, "Raw syntax error caught at Tier 0")
        self.assertEqual(funnel["tier_1_crude_bug_screened"], 1, "Raw crude bug caught at Tier 1")
        self.assertEqual(funnel["tier_2_mutation_screened"], 1, "Raw subtle boundary bug caught at Tier 2")
        self.assertEqual(funnel["tier_3_qualified"], 1, "Only conforming PR reached Tier 3")
        self.assertEqual(funnel["knapsack_admitted"], 1)
        self.assertEqual(report.admitted_candidates[0]["candidate_id"], "PR-RAW-PASS")

    def test_swebench_benchmark_runner(self):
        """Benchmark Test: Validates benchmark runner function and report persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            out_path = f.name
        try:
            summary = run_swebench_benchmark(queue_capacity=3, docker=False, output_path=out_path, verbose=False)
            self.assertEqual(summary["total_incoming_prs"], 15)
            self.assertEqual(len(summary["admitted_pull_requests"]), 3)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f_out:
                loaded = json.load(f_out)
            self.assertEqual(loaded["funnel"]["tier_3_qualified"], 7)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
