"""Unit and regression tests for Road A (Tiered Verification Gate)."""
import json
import os
import tempfile
import unittest

from governed_agent import (
    TieredVerificationGate,
    GateReceipt,
    GateBatchReport,
    CandidateSubmission,
    Governor,
    KnapsackController,
)
from governed_agent.cli import main


class TestTieredVerificationGate(unittest.TestCase):

    def setUp(self):
        self.gate = TieredVerificationGate(dry_run=True)

    def test_tier0_syntax_short_circuit_bypasses_docker(self):
        """Tier 0: Syntax error must be rejected on host with zero cost and no Docker."""
        broken_code = "def broken(x):\n    if x > 0\n        return x\n"
        receipt = self.gate.verify_candidate(
            candidate_id="PR-SYNTAX-ERR",
            candidate_code=broken_code,
        )
        self.assertFalse(receipt.passed)
        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 0)
        self.assertTrue(receipt.short_circuited)
        self.assertFalse(receipt.docker_invoked)
        self.assertEqual(receipt.total_cost, 0.0000)
        self.assertEqual(receipt.cost_savings_pct, 100.0)
        self.assertEqual(receipt.decision, "HALT_AND_REJECT")
        self.assertEqual(receipt.final_belief, 0.0)

    def test_tier1_crude_bug_short_circuit_bypasses_docker(self):
        """Tier 1: Crude logic crash on host invariant probe halts before Docker."""
        crude_code = "def calc(x):\n    return 10 // x\n"
        ref_fn = lambda x: 10 // x if x != 0 else 0

        probes = [
            {"name": "zero_div_check", "input": {"x": 0}, "tier": 1},
            {"name": "nonzero_check", "input": {"x": 2}, "tier": 1},
        ]
        receipt = self.gate.verify_candidate(
            candidate_id="PR-CRUDE-BUG",
            candidate_code=crude_code,
            reference_fn=ref_fn,
            probes=probes,
        )
        self.assertFalse(receipt.passed)
        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 1)
        self.assertTrue(receipt.short_circuited)
        self.assertFalse(receipt.docker_invoked)
        self.assertGreater(receipt.total_cost, 0.0)
        self.assertLess(receipt.total_cost, 0.001)
        self.assertEqual(receipt.decision, "HALT_AND_REJECT")

    def test_tier2_mutation_oracle_short_circuit_bypasses_docker(self):
        """Tier 2: Subtle off-by-one / boundary defect caught by mutation oracle halts before Docker."""
        # Candidate has off-by-one boundary
        subtle_code = "def is_adult(age):\n    return age > 18\n"
        # Reference requires >= 18
        ref_fn = lambda age: age >= 18

        tier1_probes = [
            {"name": "young_age", "input": {"age": 10}},
            {"name": "old_age", "input": {"age": 30}},
        ]
        tier2_probes = [
            {"name": "exact_boundary_18", "input": {"age": 18}},
        ]
        receipt = self.gate.verify_candidate(
            candidate_id="PR-SUBTLE-BUG",
            candidate_code=subtle_code,
            reference_fn=ref_fn,
            tier1_probes=tier1_probes,
            tier2_probes=tier2_probes,
        )
        self.assertFalse(receipt.passed)
        self.assertEqual(receipt.terminal_tier, 2)
        self.assertTrue(receipt.short_circuited)
        self.assertFalse(receipt.docker_invoked)
        self.assertEqual(receipt.decision, "HALT_AND_REJECT")

    def test_tier3_qualification_escalates_to_docker(self):
        """Tier 3: Conforming code passes Tiers 0-2 and escalates to Tier 3 container replay."""
        clean_code = "def add(a, b):\n    return a + b\n"
        ref_fn = lambda a=0, b=0: a + b

        t1 = [{"name": "basic_add", "input": {"a": 1, "b": 2}}]
        t2 = [
            {"name": "zero_add", "input": {"a": 0, "b": 0}},
            {"name": "large_add", "input": {"a": 100, "b": 200}},
        ]
        t3 = [{"name": "deep_container_replay", "input": {"a": -5, "b": 10}}]

        receipt = self.gate.verify_candidate(
            candidate_id="PR-CONFORMING",
            candidate_code=clean_code,
            reference_fn=ref_fn,
            tier1_probes=t1,
            tier2_probes=t2,
            tier3_probes=t3,
        )
        self.assertTrue(receipt.passed)
        self.assertTrue(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 3)
        self.assertFalse(receipt.short_circuited)
        self.assertTrue(receipt.docker_invoked)
        self.assertEqual(receipt.decision, "HALT_AND_COMMIT")
        self.assertGreaterEqual(receipt.final_belief, receipt.p_star)

    def test_batch_gate_delivers_70pct_compute_savings(self):
        """Batch Gatekeeper: Simulates SWE-bench style PR pool; validates >70% compute savings."""
        qual_probes = [
            {"name": "p0", "input": {"x": 1}, "tier": 1},
            {"name": "p1", "input": {"x": 1}, "tier": 2},
            {"name": "p2", "input": {"x": 1}, "tier": 2},
            {"name": "p3", "input": {"x": 1}, "tier": 3},
        ]
        qual_str_probes = [
            {"name": "p0", "input": {"x": ""}, "tier": 1},
            {"name": "p1", "input": {"x": ""}, "tier": 2},
            {"name": "p2", "input": {"x": ""}, "tier": 2},
            {"name": "p3", "input": {"x": ""}, "tier": 3},
        ]
        pr_batch = [
            # Conforming PRs (pass Tiers 0-3)
            {"pr_id": "PR-01", "patch_code": "def f(x):\n    return x + 1\n", "candidate_fn": lambda x=0: x+1, "reference_fn": lambda x=0: x+1, "expected_utility": 0.040, "probes": qual_probes},
            {"pr_id": "PR-02", "patch_code": "def f(x):\n    return x * 2\n", "candidate_fn": lambda x=0: x*2, "reference_fn": lambda x=0: x*2, "expected_utility": 0.035, "probes": qual_probes},
            {"pr_id": "PR-03", "patch_code": "def f(x):\n    return len(x)\n", "candidate_fn": lambda x="": len(x), "reference_fn": lambda x="": len(x), "expected_utility": 0.030, "probes": qual_str_probes},
            {"pr_id": "PR-04", "patch_code": "def f(x):\n    return bool(x)\n", "candidate_fn": lambda x=0: bool(x), "reference_fn": lambda x=0: bool(x), "expected_utility": 0.015, "probes": qual_probes},
            # Syntax broken PRs (halt at Tier 0)
            {"pr_id": "PR-05", "has_syntax_error": True},
            {"pr_id": "PR-06", "has_syntax_error": True},
            {"pr_id": "PR-07", "has_syntax_error": True},
            # Crude bug PRs (halt at Tier 1)
            {"pr_id": "PR-08", "has_crude_bug": True},
            {"pr_id": "PR-09", "has_crude_bug": True},
            {"pr_id": "PR-10", "has_crude_bug": True},
            # Subtle bug PRs (halt at Tier 2)
            {"pr_id": "PR-11", "has_subtle_bug": True},
            {"pr_id": "PR-12", "has_subtle_bug": True},
        ]

        batch_report = self.gate.gate_batch(pr_batch, capacity_K=3, review_cost_k=1)
        funnel = batch_report.funnel
        econ = batch_report.economics

        # Funnel assertions
        self.assertEqual(funnel["total_incoming_candidates"], 12)
        self.assertEqual(funnel["tier_0_syntax_screened"], 3)
        self.assertEqual(funnel["tier_1_crude_bug_screened"], 3)
        self.assertEqual(funnel["tier_2_mutation_screened"], 2)
        self.assertEqual(funnel["tier_3_qualified"], 4)
        self.assertEqual(funnel["knapsack_admitted"], 3)
        self.assertEqual(funnel["knapsack_deprioritized"], 1)

        # Economic assertions: verify >70% savings and >70% Docker runs avoided
        self.assertGreater(econ["cost_savings_pct"], 70.0)
        self.assertGreater(econ["docker_invocations_avoided_pct"], 60.0)
        self.assertEqual(econ["docker_invocations_avoided"], 8)
        self.assertGreaterEqual(econ["capacity_shadow_price_lambda"], 0.0)

    def test_cli_gate_single_file(self):
        """CLI: Run governed-agent gate on a single python file with probes."""
        cand_code = "def dummy(x):\n    return x + 1\n"
        probes = [
            {"name": f"p{i}", "input": {"x": i}, "expected": i + 1}
            for i in range(4)
        ]
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f_cand:
            f_cand.write(cand_code)
            cand_path = f_cand.name
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f_prob:
            json.dump(probes, f_prob)
            prob_path = f_prob.name
        try:
            ret = main(["gate", cand_path, "--probes", prob_path])
            self.assertEqual(ret, 0)
        finally:
            if os.path.exists(cand_path):
                os.remove(cand_path)
            if os.path.exists(prob_path):
                os.remove(prob_path)

    def test_cli_gate_batch_json(self):
        """CLI: Run governed-agent gate on a batch JSON file."""
        prs = [
            {"candidate_id": "PR-T1", "patch_code": "def f(x):\n    return x\n", "expected_utility": 0.02},
            {"candidate_id": "PR-T2", "has_syntax_error": True},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            json.dump(prs, f)
            batch_path = f.name
        try:
            ret = main(["gate", batch_path, "--capacity", "2"])
            self.assertEqual(ret, 0)
        finally:
            if os.path.exists(batch_path):
                os.remove(batch_path)

    def test_live_docker_isolation_when_available(self):
        """Tier 3: Live Docker container isolation executes successfully when Docker daemon is available."""
        from governed_agent.isolation import ProcessIsolatedRunner
        try:
            runner = ProcessIsolatedRunner()
            runner._preflight()
        except Exception as e:
            self.skipTest(f"Docker environment not ready: {e}")

        gov = Governor(reward=0.05, loss=0.05, prior=0.70, max_stages=4)
        gate = TieredVerificationGate(dry_run=False, isolated_runner=runner, governor=gov)
        code = "def solve(x):\n    return x * 2\n"
        probes = [
            {"name": "p1", "input": {"x": 1}, "tier": 1},
            {"name": "p2", "input": {"x": 5}, "tier": 2},
            {"name": "p3", "input": {"x": 10}, "tier": 3},
        ]
        receipt = gate.verify_candidate("PR-LIVE-DOCKER", code, reference_fn=lambda x: x * 2, probes=probes)
        self.assertTrue(receipt.passed)
        self.assertTrue(receipt.docker_invoked)
        self.assertEqual(receipt.terminal_tier, 3)
        self.assertGreater(receipt.total_cost, 0.0)


if __name__ == "__main__":
    unittest.main()

