"""Tests for Heterogeneous Review Costs & The c-mu Rule (Pillar V / Section 4.3).

Validates:
1. Priority inversion resolution: high-density quick fixes prioritized over low-density deep refactors
2. Strict knapsack capacity bounding under fractional hourly budgets (sum(k_i) <= K)
3. Accurate accounting of total_admitted_cost and remaining_capacity
4. Marginal density shadow price lambda_K under heterogeneous constraints
5. Cross-language bit-for-bit parity between Python reference and Rust core
6. Validation of strictly positive review costs (k_i > 0)
"""
import math
import random
import unittest

from governed_agent.knapsack import (
    CandidateSubmission as PyCandidate,
    KnapsackAdmissionReport as PyReport,
    KnapsackController as PyKnapsack,
)

try:
    from governed_agent.governed_agent_core import (
        CandidateSubmission as RustCandidate,
        KnapsackAdmissionReport as RustReport,
        KnapsackController as RustKnapsack,
    )
    HAS_RUST_CORE = True
except (ImportError, OSError):
    RustCandidate = None
    RustReport = None
    RustKnapsack = None
    HAS_RUST_CORE = False


class TestHeterogeneousQueues(unittest.TestCase):
    """Verifies heterogeneous review costs and c-mu scheduling under bounded absorption capacity."""

    def test_c_mu_rule_priority_inversion(self):
        """High-density quick fixes must be admitted before low-density deep refactors."""
        # Candidate A: 10-minute quick fix (0.167 hr), E[U] = 0.008 -> rho = 0.0479
        cand_quick = PyCandidate(
            candidate_id="PR_QUICK_FIX",
            task_id="task_1",
            posterior_belief=0.90,
            reward=0.02,
            loss=0.10,
            review_cost=1.0 / 6.0,
        )

        # Candidate B: 3.0-hour architectural refactor, higher raw utility (0.014), but rho = 0.00467
        cand_deep = PyCandidate(
            candidate_id="PR_DEEP_REFACTOR",
            task_id="task_2",
            posterior_belief=0.95,
            reward=0.02,
            loss=0.10,
            review_cost=3.0,
        )

        # Candidate C: 30-minute bug fix (0.5 hr), E[U] = 0.010 -> rho = 0.0200
        cand_med = PyCandidate(
            candidate_id="PR_MED_FEATURE",
            task_id="task_3",
            posterior_belief=0.9167,
            reward=0.02,
            loss=0.10,
            review_cost=0.5,
        )

        # Verify density ranking
        self.assertGreater(cand_deep.expected_utility, cand_quick.expected_utility)
        self.assertGreater(cand_quick.density, cand_med.density)
        self.assertGreater(cand_med.density, cand_deep.density)
        self.assertEqual(cand_quick.density, cand_quick.c_mu_index)

        # Controller with 1.0 hour budget
        controller = PyKnapsack(default_capacity=1.0)
        report = controller.admit_batch([cand_deep, cand_quick, cand_med])

        # Deep refactor (3.0 hr) cannot fit in 1.0 hr budget
        # Quick fix (0.167 hr) and Medium fix (0.5 hr) fit: total cost = 0.667 hr <= 1.0 hr
        admitted_ids = [c.candidate_id for c in report.admitted]
        self.assertIn("PR_QUICK_FIX", admitted_ids)
        self.assertIn("PR_MED_FEATURE", admitted_ids)
        self.assertNotIn("PR_DEEP_REFACTOR", admitted_ids)

        self.assertAlmostEqual(report.total_admitted_cost, (1.0 / 6.0) + 0.5, places=5)
        self.assertAlmostEqual(report.remaining_capacity, 1.0 - report.total_admitted_cost, places=5)
        self.assertAlmostEqual(report.shadow_price_lambda, cand_deep.density, places=5)

    def test_fractional_hourly_capacity_packing(self):
        """Multiple candidates of varying costs packed up to fractional capacity budget K."""
        controller = PyKnapsack(default_capacity=5.5)

        candidates = [
            PyCandidate(candidate_id=f"c_{i}", task_id="t", posterior_belief=0.88, reward=0.02, loss=0.10, review_cost=c)
            for i, c in enumerate([1.5, 2.0, 1.0, 1.5, 0.5, 3.0])
        ]

        report = controller.admit_batch(candidates)
        self.assertLessEqual(report.total_admitted_cost, 5.5 + 1e-9)
        self.assertGreater(report.total_admitted, 0)
        self.assertEqual(report.total_candidates, 6)
        self.assertEqual(report.total_admitted + report.total_rejected, 6)

    def test_invalid_review_costs_raise(self):
        """Non-positive review costs must be rejected with ValueError."""
        with self.assertRaises(ValueError):
            PyCandidate(candidate_id="c", task_id="t", posterior_belief=0.9, review_cost=0.0)

        with self.assertRaises(ValueError):
            PyCandidate(candidate_id="c", task_id="t", posterior_belief=0.9, review_cost=-1.5)

        if HAS_RUST_CORE:
            with self.assertRaises(ValueError):
                RustCandidate(candidate_id="c", task_id="t", posterior_belief=0.9, review_cost=0.0)
            with self.assertRaises(ValueError):
                RustCandidate(candidate_id="c", task_id="t", posterior_belief=0.9, review_cost=-1.0)

    @unittest.skipUnless(HAS_RUST_CORE, "Rust core required for cross-language parity")
    def test_cross_language_parity_heterogeneous(self):
        """Python reference and Rust core must yield identical decisions and accounting."""
        rng = random.Random(42)

        for scenario_idx in range(30):
            capacity = rng.uniform(2.0, 15.0)
            n_cands = rng.randint(5, 35)

            cands_meta = [
                (
                    f"c_{j}",
                    rng.uniform(0.70, 0.99),
                    rng.uniform(0.01, 0.05),
                    rng.uniform(0.05, 0.15),
                    rng.uniform(0.2, 3.5),
                )
                for j in range(n_cands)
            ]

            py_cands = [
                PyCandidate(candidate_id=cid, task_id="t", posterior_belief=p, reward=r, loss=l, review_cost=cost)
                for cid, p, r, l, cost in cands_meta
            ]
            rust_cands = [
                RustCandidate(candidate_id=cid, task_id="t", posterior_belief=p, reward=r, loss=l, review_cost=cost)
                for cid, p, r, l, cost in cands_meta
            ]

            py_ks = PyKnapsack(default_capacity=capacity)
            rust_ks = RustKnapsack(default_capacity=capacity)

            py_rep = py_ks.admit_batch(py_cands)
            rust_rep = rust_ks.admit_batch(rust_cands)

            self.assertEqual(py_rep.total_candidates, rust_rep.total_candidates)
            self.assertEqual(py_rep.total_admitted, rust_rep.total_admitted)
            self.assertEqual(py_rep.total_rejected, rust_rep.total_rejected)
            self.assertAlmostEqual(py_rep.total_welfare, rust_rep.total_welfare, delta=1e-7)
            self.assertAlmostEqual(py_rep.total_admitted_cost, rust_rep.total_admitted_cost, delta=1e-7)
            self.assertAlmostEqual(py_rep.remaining_capacity, rust_rep.remaining_capacity, delta=1e-7)
            self.assertAlmostEqual(py_rep.shadow_price_lambda, rust_rep.shadow_price_lambda, delta=1e-7)

            py_admitted_ids = [c.candidate_id for c in py_rep.admitted]
            rust_admitted_ids = [c.candidate_id for c in rust_rep.admitted]
            self.assertEqual(py_admitted_ids, rust_admitted_ids)

    def test_density_greedy_outperforms_fifo_when_head_of_line_blocks(self):
        """Demonstrates that density-greedy avoids head-of-line blocking by expensive marginal candidates."""
        candidates = [
            # High cost, marginal utility
            PyCandidate(candidate_id="c_heavy", task_id="t", posterior_belief=0.85, reward=0.02, loss=0.10, review_cost=4.0),
            # Low cost, high utility
            PyCandidate(candidate_id="c_fast_1", task_id="t", posterior_belief=0.95, reward=0.02, loss=0.10, review_cost=0.5),
            # Low cost, high utility
            PyCandidate(candidate_id="c_fast_2", task_id="t", posterior_belief=0.92, reward=0.02, loss=0.10, review_cost=0.5),
        ]

        # Capacity K = 4.0
        # Naive FIFO admits c_heavy first (cost 4.0), exhausting entire budget K!
        # Welfare(FIFO) = E[U(c_heavy)] = 0.85*0.02 - 0.15*0.10 = 0.017 - 0.015 = 0.002
        fifo_welfare = candidates[0].expected_utility

        # c-mu density knapsack prioritizes fast fixes
        controller = PyKnapsack(default_capacity=4.0)
        report = controller.admit_batch(candidates)

        cmu_welfare = report.total_welfare
        self.assertGreater(cmu_welfare, fifo_welfare)
        admitted_ids = [c.candidate_id for c in report.admitted]
        self.assertIn("c_fast_1", admitted_ids)
        self.assertIn("c_fast_2", admitted_ids)

    def test_density_greedy_heuristic_can_underperform_fifo_due_to_indivisibility(self):
        """Density-greedy is a heuristic for discrete 0-1 knapsack, not an unconditional welfare guarantee.

        Because discrete jobs are indivisible, greedily packing items by descending value density
        rho_i = E[U_i] / k_i can leave stranded capacity that could have been filled by a larger
        job with higher total expected utility.

        Counterexample:
        - Candidate 1 (FIFO head): cost = 5.0, utility = 0.100 (rho = 0.020)
        - Candidate 2: cost = 3.0, utility = 0.066 (rho = 0.022)
        - Candidate 3: cost = 3.0, utility = 0.010 (rho = 0.0033)

        Under Capacity K = 5.0:
        - FIFO admits Candidate 1 (cost 5.0), total welfare = 0.100.
        - Density-greedy admits Candidate 2 (cost 3.0, rho=0.022), leaving 2.0 units stranded.
          Neither Candidate 1 (5.0) nor Candidate 3 (3.0) can fit in the remaining 2.0 units.
          Density-greedy total welfare = 0.066 < FIFO welfare (0.100).
        """
        c1 = PyCandidate(candidate_id="c_fifo_head", task_id="t", posterior_belief=1.0, reward=0.10, loss=0.10, review_cost=5.0)
        c2 = PyCandidate(candidate_id="c_high_density", task_id="t", posterior_belief=1.0, reward=0.066, loss=0.10, review_cost=3.0)
        c3 = PyCandidate(candidate_id="c_low_density", task_id="t", posterior_belief=1.0, reward=0.010, loss=0.10, review_cost=3.0)

        # 1. FIFO: admits Candidate 1, exhausting K=5.0
        fifo_welfare = c1.expected_utility
        self.assertAlmostEqual(fifo_welfare, 0.100, places=3)

        # 2. Density-Greedy Knapsack
        controller = PyKnapsack(default_capacity=5.0)
        report = controller.admit_batch([c1, c2, c3])

        # Density ranking: c2 (0.022) > c1 (0.020) > c3 (0.0033)
        # c2 is admitted (cost 3.0). Remaining capacity = 2.0.
        # Neither c1 nor c3 fits in 2.0.
        self.assertEqual(report.total_admitted, 1)
        self.assertEqual(report.admitted[0].candidate_id, "c_high_density")
        self.assertAlmostEqual(report.total_welfare, 0.066, places=3)
        self.assertAlmostEqual(report.remaining_capacity, 2.0, places=3)

        # Confirms: FIFO (0.100) > Density-Greedy (0.066) due to discrete indivisibility
        self.assertGreater(fifo_welfare, report.total_welfare)

    def test_global_review_cost_k_backward_compatibility(self):
        """When candidates omit review_cost, they must inherit the controller's global review_cost_k."""
        cands = [
            PyCandidate(candidate_id="c1", task_id="t", posterior_belief=0.99, reward=0.02, loss=0.10),
            PyCandidate(candidate_id="c2", task_id="t", posterior_belief=0.95, reward=0.02, loss=0.10),
            PyCandidate(candidate_id="c3", task_id="t", posterior_belief=0.90, reward=0.02, loss=0.10),
        ]

        # Capacity K = 2, Global review_cost_k = 2 -> m* = floor(2 / 2) = 1
        controller = PyKnapsack(default_capacity=2, review_cost_k=2)
        report = controller.admit_batch(cands)

        self.assertEqual(report.total_admitted, 1)
        self.assertEqual(report.optimal_selection_depth, 1)
        self.assertEqual(report.total_admitted_cost, 2.0)
        self.assertEqual(report.remaining_capacity, 0.0)
        self.assertEqual(report.admitted[0].candidate_id, "c1")

        if HAS_RUST_CORE:
            rust_cands = [
                RustCandidate(candidate_id="c1", task_id="t", posterior_belief=0.99, reward=0.02, loss=0.10),
                RustCandidate(candidate_id="c2", task_id="t", posterior_belief=0.95, reward=0.02, loss=0.10),
                RustCandidate(candidate_id="c3", task_id="t", posterior_belief=0.90, reward=0.02, loss=0.10),
            ]
            rust_controller = RustKnapsack(default_capacity=2.0, review_cost_k=2.0)
            rust_report = rust_controller.admit_batch(rust_cands)

            self.assertEqual(rust_report.total_admitted, 1)
            self.assertEqual(rust_report.optimal_selection_depth, 1)
            self.assertEqual(rust_report.total_admitted_cost, 2.0)
            self.assertEqual(rust_report.remaining_capacity, 0.0)
            self.assertEqual(rust_report.admitted[0].candidate_id, "c1")


if __name__ == "__main__":
    unittest.main()
