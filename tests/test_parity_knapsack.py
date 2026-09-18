"""Cross-Language Parity Tests for Pillar V: Knapsack Queue Admission.

Verifies bit-for-bit equivalence between the native Rust core (`governed_agent_core.KnapsackController`)
and the pure-Python reference implementation (`governed_agent.knapsack.KnapsackController`).
"""
import math
import random
import unittest

try:
    from governed_agent.governed_agent_core import (
        CandidateSubmission as RustCandidate,
        KnapsackController as RustKnapsack,
    )
    HAS_RUST_CORE = True
except (ImportError, OSError):
    RustCandidate = None
    RustKnapsack = None
    HAS_RUST_CORE = False

from governed_agent.knapsack import (
    CandidateSubmission as PyCandidate,
    KnapsackController as PyKnapsack,
)


@unittest.skipUnless(HAS_RUST_CORE, "Native Rust core (governed_agent_core) unavailable")
class TestKnapsackParity(unittest.TestCase):
    """Equivalence tests between Python and Rust Knapsack admission engines."""

    def assert_float_almost_equal(self, a: float, b: float, tol: float = 1e-10, msg: str = ""):
        self.assertTrue(
            math.isclose(a, b, abs_tol=tol, rel_tol=tol),
            f"{msg}: {a} != {b} (diff={abs(a - b):.2e}, tol={tol:.2e})"
        )

    def test_default_initialization(self):
        py_ks = PyKnapsack(default_capacity=15, review_cost_k=2)
        rust_ks = RustKnapsack(default_capacity=15, review_cost_k=2)

        self.assertEqual(py_ks.capacity_K, rust_ks.capacity_K)
        self.assertEqual(py_ks.review_cost_k, rust_ks.review_cost_k)

    def test_empty_candidates_batch(self):
        py_ks = PyKnapsack()
        rust_ks = RustKnapsack()

        py_rep = py_ks.admit_batch([])
        rust_rep = rust_ks.admit_batch([])

        self.assertEqual(py_rep.total_candidates, rust_rep.total_candidates)
        self.assertEqual(py_rep.total_admitted, rust_rep.total_admitted)
        self.assertEqual(py_rep.total_rejected, rust_rep.total_rejected)
        self.assertEqual(py_rep.optimal_selection_depth, rust_rep.optimal_selection_depth)
        self.assert_float_almost_equal(py_rep.total_welfare, rust_rep.total_welfare)
        self.assert_float_almost_equal(py_rep.shadow_price_lambda, rust_rep.shadow_price_lambda)

    def test_deterministic_candidate_scenarios(self):
        scenarios = [
            # Scenario A: All candidates acceptable, capacity binds (N > K)
            {
                "capacity": 3,
                "review_cost": 1,
                "posteriors": [0.95, 0.92, 0.88, 0.85, 0.84],
            },
            # Scenario B: Mixed acceptability, some below p* (p* = 0.10 / 0.12 = 0.8333)
            {
                "capacity": 5,
                "review_cost": 1,
                "posteriors": [0.99, 0.90, 0.70, 0.50, 0.30],
            },
            # Scenario C: Non-unit review cost k=2, capacity K=5 -> floor(5/2) = 2
            {
                "capacity": 5,
                "review_cost": 2,
                "posteriors": [0.95, 0.90, 0.88, 0.86, 0.85],
            },
            # Scenario D: Fewer candidates than capacity (N < K)
            {
                "capacity": 10,
                "review_cost": 1,
                "posteriors": [0.95, 0.90],
            },
            # Scenario E: None acceptable
            {
                "capacity": 10,
                "review_cost": 1,
                "posteriors": [0.50, 0.40, 0.30],
            },
        ]

        for s in scenarios:
            with self.subTest(scenario=s):
                py_ks = PyKnapsack(default_capacity=s["capacity"], review_cost_k=s["review_cost"])
                rust_ks = RustKnapsack(default_capacity=s["capacity"], review_cost_k=s["review_cost"])

                py_cands = [
                    PyCandidate(
                        candidate_id=f"c_{i}",
                        task_id="t1",
                        posterior_belief=p,
                        reward=0.02,
                        loss=0.10,
                    )
                    for i, p in enumerate(s["posteriors"])
                ]

                rust_cands = [
                    RustCandidate(
                        candidate_id=f"c_{i}",
                        task_id="t1",
                        posterior_belief=p,
                        reward=0.02,
                        loss=0.10,
                    )
                    for i, p in enumerate(s["posteriors"])
                ]

                py_rep = py_ks.admit_batch(py_cands)
                rust_rep = rust_ks.admit_batch(rust_cands)

                self.assertEqual(py_rep.capacity_K, rust_rep.capacity_K)
                self.assertEqual(py_rep.review_cost_k, rust_rep.review_cost_k)
                self.assertEqual(py_rep.optimal_selection_depth, rust_rep.optimal_selection_depth)
                self.assertEqual(py_rep.total_candidates, rust_rep.total_candidates)
                self.assertEqual(py_rep.total_admitted, rust_rep.total_admitted)
                self.assertEqual(py_rep.total_rejected, rust_rep.total_rejected)
                self.assert_float_almost_equal(py_rep.total_welfare, rust_rep.total_welfare)
                self.assert_float_almost_equal(py_rep.shadow_price_lambda, rust_rep.shadow_price_lambda)

                py_admitted_ids = [c.candidate_id for c in py_rep.admitted]
                rust_admitted_ids = [c.candidate_id for c in rust_rep.admitted]
                self.assertEqual(py_admitted_ids, rust_admitted_ids)

                py_rejected_ids = [c.candidate_id for c in py_rep.rejected]
                rust_rejected_ids = [c.candidate_id for c in rust_rep.rejected]
                self.assertEqual(py_rejected_ids, rust_rejected_ids)

    def test_randomized_monte_carlo_parity(self):
        """Randomized batch verification under randomized capacities and costs."""
        rng = random.Random(42)
        for _ in range(25):
            cap = rng.randint(1, 20)
            cost = rng.randint(1, 4)
            n_cands = rng.randint(0, 40)

            py_ks = PyKnapsack(default_capacity=cap, review_cost_k=cost)
            rust_ks = RustKnapsack(default_capacity=cap, review_cost_k=cost)

            cands_data = [
                (f"cand_{j}", rng.uniform(0.1, 0.99), rng.uniform(0.01, 0.05), rng.uniform(0.05, 0.20))
                for j in range(n_cands)
            ]

            py_cands = [
                PyCandidate(candidate_id=cid, task_id="rnd", posterior_belief=p, reward=r, loss=l)
                for cid, p, r, l in cands_data
            ]
            rust_cands = [
                RustCandidate(candidate_id=cid, task_id="rnd", posterior_belief=p, reward=r, loss=l)
                for cid, p, r, l in cands_data
            ]

            # Also test optional overrides for capacity and cost
            override_cap = rng.choice([None, rng.randint(1, 30)])
            override_cost = rng.choice([None, rng.randint(1, 5)])

            py_rep = py_ks.admit_batch(py_cands, capacity_K=override_cap, review_cost_k=override_cost)
            rust_rep = rust_ks.admit_batch(rust_cands, capacity_K=override_cap, review_cost_k=override_cost)

            self.assertEqual(py_rep.total_candidates, rust_rep.total_candidates)
            self.assertEqual(py_rep.total_admitted, rust_rep.total_admitted)
            self.assertEqual(py_rep.total_rejected, rust_rep.total_rejected)
            self.assertEqual(py_rep.optimal_selection_depth, rust_rep.optimal_selection_depth)
            self.assert_float_almost_equal(py_rep.total_welfare, rust_rep.total_welfare)
            self.assert_float_almost_equal(py_rep.shadow_price_lambda, rust_rep.shadow_price_lambda)

    def test_heterogeneous_monte_carlo_parity(self):
        """Randomized batch verification under heterogeneous review costs and fractional budgets."""
        rng = random.Random(999)
        for _ in range(25):
            cap = rng.uniform(1.5, 25.0)
            n_cands = rng.randint(0, 35)

            py_ks = PyKnapsack(default_capacity=cap)
            rust_ks = RustKnapsack(default_capacity=cap)

            cands_data = [
                (
                    f"cand_{j}",
                    rng.uniform(0.1, 0.99),
                    rng.uniform(0.01, 0.05),
                    rng.uniform(0.05, 0.20),
                    rng.uniform(0.1, 4.0),
                )
                for j in range(n_cands)
            ]

            py_cands = [
                PyCandidate(candidate_id=cid, task_id="rnd_het", posterior_belief=p, reward=r, loss=l, review_cost=c)
                for cid, p, r, l, c in cands_data
            ]
            rust_cands = [
                RustCandidate(candidate_id=cid, task_id="rnd_het", posterior_belief=p, reward=r, loss=l, review_cost=c)
                for cid, p, r, l, c in cands_data
            ]

            py_rep = py_ks.admit_batch(py_cands)
            rust_rep = rust_ks.admit_batch(rust_cands)

            self.assertEqual(py_rep.total_candidates, rust_rep.total_candidates)
            self.assertEqual(py_rep.total_admitted, rust_rep.total_admitted)
            self.assertEqual(py_rep.total_rejected, rust_rep.total_rejected)
            self.assert_float_almost_equal(py_rep.total_welfare, rust_rep.total_welfare)
            self.assert_float_almost_equal(py_rep.total_admitted_cost, rust_rep.total_admitted_cost)
            self.assert_float_almost_equal(py_rep.remaining_capacity, rust_rep.remaining_capacity)
            self.assert_float_almost_equal(py_rep.shadow_price_lambda, rust_rep.shadow_price_lambda)

            py_admitted_ids = [c.candidate_id for c in py_rep.admitted]
            rust_admitted_ids = [c.candidate_id for c in rust_rep.admitted]
            self.assertEqual(py_admitted_ids, rust_admitted_ids)

    def test_invalid_knapsack_arguments(self):
        with self.assertRaises(ValueError):
            PyKnapsack(default_capacity=0)
        with self.assertRaises(ValueError):
            RustKnapsack(default_capacity=0)

        with self.assertRaises(ValueError):
            PyKnapsack(review_cost_k=0)
        with self.assertRaises(ValueError):
            RustKnapsack(review_cost_k=0)

        py_ks = PyKnapsack()
        rust_ks = RustKnapsack()
        for override in ({"capacity_K": 0}, {"review_cost_k": 0}):
            with self.assertRaises(ValueError):
                py_ks.admit_batch([], **override)
            with self.assertRaises(ValueError):
                rust_ks.admit_batch([], **override)


if __name__ == "__main__":
    unittest.main()
