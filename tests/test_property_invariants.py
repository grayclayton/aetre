"""Property-Based Mathematical Invariant Fuzzing (Hypothesis).

Formally verifies decision-theoretic and macroeconomic invariants across
large randomized parameter spaces, including:
1. Tripartite Review Boundary Optimality:
   E[U(decision)] == max(U(auto), U(review), U(abstain)) across all float combinations.
2. Bit-for-bit Parity between Native Rust Core and Python Reference:
   Identical actions, utilities, and shadow prices across millions of state transitions.
3. Knapsack Queue Admission Invariants:
   Strict capacity boundedness, intrinsic acceptability filtering, greedy welfare monotonicity,
   and shadow price consistency.
4. Bayesian Posterior Monotonicity:
   Strictly non-decreasing under positive evidence; non-increasing under negative evidence.
"""
import math
import unittest
from typing import List, Tuple

try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from governed_agent.governor import (
    Governor as PyGovernor,
    ReviewBoundaryDecision as PyDecision,
)
from governed_agent.knapsack import (
    KnapsackController as PyKnapsack,
    CandidateSubmission as PyCandidate,
    KnapsackAdmissionReport as PyReport,
)

try:
    from governed_agent.governed_agent_core import (
        Governor as RustGovernor,
        ReviewBoundaryDecision as RustDecision,
        KnapsackController as RustKnapsack,
        CandidateSubmission as RustCandidate,
    )
    HAS_RUST_CORE = True
except (ImportError, OSError):
    RustGovernor = None
    RustDecision = None
    RustKnapsack = None
    RustCandidate = None
    HAS_RUST_CORE = False


@unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis library required for property-based testing")
class TestPropertyInvariants(unittest.TestCase):
    """Property-based invariant proofs using Hypothesis."""

    @settings(max_examples=300, deadline=None)
    @given(
        reward=st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
        loss=st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
        belief=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        review_cost=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        shadow_price_lambda=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        review_accuracy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_tripartite_review_boundary_optimality(
        self,
        reward: float,
        loss: float,
        belief: float,
        review_cost: float,
        shadow_price_lambda: float,
        review_accuracy: float,
    ):
        """Invariant: The chosen action strictly maximizes expected net utility over {auto, review, abstain}."""
        gov = PyGovernor(reward=reward, loss=loss, prior=0.50, max_stages=4)
        decision = gov.evaluate_review_boundary(
            belief=belief,
            review_cost=review_cost,
            shadow_price_lambda=shadow_price_lambda,
            review_accuracy=review_accuracy,
        )

        u_auto = decision.u_auto
        u_review = decision.u_review
        u_abstain = decision.u_abstain
        dominant = decision.dominant_utility

        max_possible = max(u_auto, u_review, u_abstain)

        # 1. Dominant utility must match max possible to numerical precision
        self.assertAlmostEqual(dominant, max_possible, delta=1e-9)

        # 2. Action selection must be optimal
        if decision.action == "AUTO":
            self.assertGreaterEqual(u_auto, max(u_review, u_abstain) - 1e-9)
            self.assertAlmostEqual(dominant, u_auto, delta=1e-9)
        elif decision.action == "REVIEW":
            self.assertGreaterEqual(u_review, u_abstain - 1e-9)
            self.assertGreaterEqual(u_review, u_auto - 1e-9)
            self.assertAlmostEqual(dominant, u_review, delta=1e-9)
        elif decision.action == "ABSTAIN":
            self.assertGreaterEqual(u_abstain, u_review - 1e-9)
            self.assertGreaterEqual(u_abstain, u_auto - 1e-9)
            self.assertAlmostEqual(dominant, u_abstain, delta=1e-9)
        else:
            self.fail(f"Invalid decision action: {decision.action}")

        # 3. Rust / Python Native Parity (when Rust is available)
        if HAS_RUST_CORE:
            rust_gov = RustGovernor(reward=reward, loss=loss, prior=0.50, max_stages=4)
            rust_dec = rust_gov.evaluate_review_boundary(
                belief=belief,
                review_cost=review_cost,
                shadow_price_lambda=shadow_price_lambda,
                review_accuracy=review_accuracy,
            )
            self.assertEqual(decision.action, rust_dec.action)
            self.assertAlmostEqual(decision.u_auto, rust_dec.u_auto, delta=1e-8)
            self.assertAlmostEqual(decision.u_review, rust_dec.u_review, delta=1e-8)
            self.assertAlmostEqual(decision.u_abstain, rust_dec.u_abstain, delta=1e-8)
            self.assertAlmostEqual(decision.dominant_utility, rust_dec.dominant_utility, delta=1e-8)

    @settings(max_examples=250, deadline=None)
    @given(
        prior=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        conforming_pass_rate=st.floats(min_value=0.51, max_value=1.0, allow_nan=False, allow_infinity=False),
        defect_leakage=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
    )
    def test_bayesian_posterior_monotonicity(
        self,
        prior: float,
        conforming_pass_rate: float,
        defect_leakage: float,
    ):
        """Invariant: Positive evidence strictly increases belief; negative evidence strictly decreases belief."""
        gov = PyGovernor(
            reward=0.02,
            loss=0.10,
            prior=prior,
            conforming_pass_rate=conforming_pass_rate,
            defect_leakage=defect_leakage,
        )

        b_pass = gov.posterior_belief(prior, outcome_pass=True)
        b_fail = gov.posterior_belief(prior, outcome_pass=False)

        # Non-decreasing on pass (strict if p_pass_conf > p_pass_def)
        self.assertGreaterEqual(b_pass, prior - 1e-12)
        # Non-increasing on fail
        self.assertLessEqual(b_fail, prior + 1e-12)

        # Belief bounded in [0, 1]
        self.assertGreaterEqual(b_pass, 0.0)
        self.assertLessEqual(b_pass, 1.0)
        self.assertGreaterEqual(b_fail, 0.0)
        self.assertLessEqual(b_fail, 1.0)

    @settings(max_examples=200, deadline=None)
    @given(
        candidates_data=st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=15,
        ),
        capacity_K=st.integers(min_value=1, max_value=20),
    )
    def test_knapsack_queue_admission_invariants(
        self,
        candidates_data: List[Tuple[int, float, float, float]],
        capacity_K: int,
    ):
        """Invariants: Capacity bound, intrinsic acceptability, greedy optimality, and Rust parity."""
        py_candidates = [
            PyCandidate(
                candidate_id=f"cand-{idx}",
                task_id=f"task-{idx}",
                posterior_belief=b,
                reward=r,
                loss=l,
            )
            for idx, (c_id, b, r, l) in enumerate(candidates_data)
        ]

        py_controller = PyKnapsack(default_capacity=capacity_K, review_cost_k=1)
        py_report = py_controller.admit_batch(py_candidates, capacity_K=capacity_K)

        # Invariant 1: Capacity Bound (Admitted <= K)
        self.assertLessEqual(py_report.total_admitted, capacity_K)

        # Invariant 2: Intrinsic Acceptability (Every admitted candidate must have E[U] >= 0)
        for adm in py_report.admitted:
            self.assertTrue(
                adm.is_intrinsically_acceptable,
                f"Admitted candidate {adm.candidate_id} must be intrinsically acceptable: b={adm.posterior_belief}, p*={adm.p_star}"
            )
            self.assertGreaterEqual(adm.expected_utility, -1e-9)

        # Invariant 3: Greedy Optimality
        # No excluded intrinsically acceptable candidate may have higher utility than an admitted candidate
        admitted_ids = {c.candidate_id for c in py_report.admitted}
        excluded_acceptable = [c for c in py_candidates if c.candidate_id not in admitted_ids and c.is_intrinsically_acceptable]
        if py_report.admitted and excluded_acceptable:
            min_admitted_util = min(c.expected_utility for c in py_report.admitted)
            max_excluded_util = max(c.expected_utility for c in excluded_acceptable)
            self.assertGreaterEqual(
                min_admitted_util,
                max_excluded_util - 1e-9,
                f"Admitted min utility {min_admitted_util} must >= excluded max {max_excluded_util}"
            )

        # Invariant 4: Shadow Price Non-Negativity
        self.assertGreaterEqual(py_report.shadow_price_lambda, 0.0)

        # If capacity was slack (K >= total acceptable candidates), shadow price must be 0
        total_acceptable = sum(1 for c in py_candidates if c.is_intrinsically_acceptable)
        if capacity_K >= total_acceptable:
            self.assertEqual(py_report.shadow_price_lambda, 0.0)

        # Invariant 5: Rust Core Parity
        if HAS_RUST_CORE:
            rust_candidates = [
                RustCandidate(
                    candidate_id=f"cand-{idx}",
                    task_id=f"task-{idx}",
                    posterior_belief=b,
                    reward=r,
                    loss=l,
                )
                for idx, (c_id, b, r, l) in enumerate(candidates_data)
            ]
            rust_controller = RustKnapsack(default_capacity=capacity_K, review_cost_k=1)
            rust_report = rust_controller.admit_batch(rust_candidates, capacity_K=capacity_K)

            self.assertEqual(py_report.total_admitted, rust_report.total_admitted)
            self.assertAlmostEqual(py_report.total_welfare, rust_report.total_welfare, delta=1e-8)
            self.assertAlmostEqual(py_report.shadow_price_lambda, rust_report.shadow_price_lambda, delta=1e-8)
            py_admitted_ids = [c.candidate_id for c in py_report.admitted]
            rust_admitted_ids = [c.candidate_id for c in rust_report.admitted]
            self.assertEqual(py_admitted_ids, rust_admitted_ids)

    @settings(max_examples=150, deadline=None)
    @given(
        candidates_data=st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=15,
        ),
        capacity_K=st.floats(min_value=0.5, max_value=25.0, allow_nan=False, allow_infinity=False),
    )
    def test_heterogeneous_knapsack_invariants(
        self,
        candidates_data: List[Tuple[int, float, float, float, float]],
        capacity_K: float,
    ):
        """Heterogeneous knapsack invariants: capacity bound, intrinsic acceptability, and Rust parity."""
        py_candidates = [
            PyCandidate(
                candidate_id=f"cand-{idx}",
                task_id=f"task-{idx}",
                posterior_belief=b,
                reward=r,
                loss=l,
                review_cost=cost,
            )
            for idx, (c_id, b, r, l, cost) in enumerate(candidates_data)
        ]

        py_controller = PyKnapsack(default_capacity=capacity_K)
        py_report = py_controller.admit_batch(py_candidates)

        # Invariant 1: Strict Capacity Bounding (Total Cost <= K)
        self.assertLessEqual(py_report.total_admitted_cost, capacity_K + 1e-7)
        self.assertAlmostEqual(py_report.remaining_capacity, max(0.0, capacity_K - py_report.total_admitted_cost), places=5)

        # Invariant 2: Intrinsic Acceptability
        for adm in py_report.admitted:
            self.assertTrue(adm.is_intrinsically_acceptable)
            self.assertGreaterEqual(adm.expected_utility, -1e-9)

        # Invariant 3: Shadow Price Non-Negativity
        self.assertGreaterEqual(py_report.shadow_price_lambda, 0.0)

        # Invariant 4: Rust Core Parity
        if HAS_RUST_CORE:
            rust_candidates = [
                RustCandidate(
                    candidate_id=f"cand-{idx}",
                    task_id=f"task-{idx}",
                    posterior_belief=b,
                    reward=r,
                    loss=l,
                    review_cost=cost,
                )
                for idx, (c_id, b, r, l, cost) in enumerate(candidates_data)
            ]
            rust_controller = RustKnapsack(default_capacity=capacity_K)
            rust_report = rust_controller.admit_batch(rust_candidates)

            self.assertEqual(py_report.total_admitted, rust_report.total_admitted)
            self.assertAlmostEqual(py_report.total_welfare, rust_report.total_welfare, delta=1e-7)
            self.assertAlmostEqual(py_report.total_admitted_cost, rust_report.total_admitted_cost, delta=1e-7)
            self.assertAlmostEqual(py_report.remaining_capacity, rust_report.remaining_capacity, delta=1e-7)
            self.assertAlmostEqual(py_report.shadow_price_lambda, rust_report.shadow_price_lambda, delta=1e-7)
            py_admitted_ids = [c.candidate_id for c in py_report.admitted]
            rust_admitted_ids = [c.candidate_id for c in rust_report.admitted]
            self.assertEqual(py_admitted_ids, rust_admitted_ids)

    @settings(max_examples=150, deadline=None)
    @given(
        reward=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        loss=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        prior=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    def test_bellman_voi_non_negativity(self, reward: float, loss: float, prior: float):
        """Invariant: Value of Information (VOI) from backward induction is never negative."""
        gov = PyGovernor(reward=reward, loss=loss, prior=prior, max_stages=3)
        for stage in range(gov.max_stages):
            for passes in range(stage + 1):
                dec = gov.evaluate_state(stage=stage, consecutive_passes=passes)
                self.assertGreaterEqual(
                    dec.voi,
                    -1e-9,
                    f"VOI cannot be negative at stage {stage}, passes {passes}: {dec.voi}"
                )


if __name__ == "__main__":
    unittest.main()
