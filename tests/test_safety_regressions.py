"""Regression tests for fail-closed execution and policy enforcement."""
import unittest

from governed_agent.governor import Governor
from governed_agent.isolation import ExecutionStatus, IsolatedExecutionResult
from governed_agent.knapsack import CandidateSubmission, KnapsackController
from governed_agent.mutator import Mutant, MutationEngine
from governed_agent.pyramid import TieredPyramid


class OfflineRunner:
    def run_candidate(self, **kwargs):
        return IsolatedExecutionResult(
            ExecutionStatus.INFRASTRUCTURE_FAILURE,
            diagnostic="Docker unavailable",
        )


class TestSafetyRegressions(unittest.TestCase):
    def test_mutant_generation_does_not_execute_module(self):
        source = "raise RuntimeError('host side effect')\ndef target(x): return x + 1"
        mutants = MutationEngine().generate_mutants(source, max_mutants=1)
        self.assertEqual(len(mutants), 1)
        self.assertIsNone(mutants[0].fn)

    def test_infrastructure_failure_cannot_qualify_mutants(self):
        mutant = Mutant("m1", "AOR", 1, "+ to -", "def target(x): return x - 1")
        report = MutationEngine().evaluate_mutation_score(
            [{"input": {"x": 2}}],
            lambda x: x + 1,
            [mutant],
            isolated_runner=OfflineRunner(),
        )
        self.assertFalse(report.is_qualified)
        self.assertEqual(report.killed_mutants, 0)
        self.assertEqual(report.inconclusive_mutants, 1)

    def test_governor_halt_prevents_tier_one_probes(self):
        governor = Governor(max_stages=1, cost_schedule=[100.0])
        report = TieredPyramid(trusted_in_process=True).run_tiered_pipeline(
            "candidate",
            "def target(): return 1",
            lambda: 1,
            lambda: 1,
            [{"input": {}}] * 3,
            [],
            governor=governor,
        )
        self.assertEqual(report.probes_executed, 1)  # syntax gate only
        self.assertEqual(report.decision, "HALT_AND_REJECT")

    def test_governor_horizon_limits_tier_one_probes(self):
        governor = Governor(
            max_stages=1,
            cost_schedule=[0.0],
            defect_leakage=0.1,
        )
        report = TieredPyramid(trusted_in_process=True).run_tiered_pipeline(
            "candidate",
            "def target(): return 1",
            lambda: 1,
            lambda: 1,
            [{"input": {}}] * 3,
            [],
            governor=governor,
        )
        self.assertEqual(report.probes_executed, 2)  # syntax + one paid probe
        self.assertTrue(report.admitted)

    def test_knapsack_overrides_are_validated(self):
        controller = KnapsackController()
        with self.assertRaises(ValueError):
            controller.admit_batch([], capacity_K=0)
        with self.assertRaises(ValueError):
            controller.admit_batch([], review_cost_k=0)

    def test_shadow_price_uses_one_capacity_unit(self):
        candidates = [
            CandidateSubmission("a", "t", 0.99),
            CandidateSubmission("b", "t", 0.95),
        ]
        controller = KnapsackController(default_capacity=2, review_cost_k=2)
        report = controller.admit_batch(candidates)
        self.assertEqual(report.total_admitted, 1)
        self.assertEqual(report.optimal_selection_depth, 1)
        self.assertEqual(report.total_admitted_cost, 2.0)
        self.assertEqual(report.shadow_price_lambda, 0.0)


if __name__ == "__main__":
    unittest.main()
