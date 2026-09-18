"""Tests for Section 4.1 Tripartite Review Boundary: Auto vs Review vs Abstain.

Formalized in Gray (2026d) The Admission Frontier:
    E[U(auto, theta) | z_t] >= max(E[U(review, theta) | z_t], E[U(abstain, theta) | z_t])
"""
from __future__ import annotations

import unittest
from governed_agent.governor import Governor as PyGovernor, ReviewBoundaryDecision as PyDecision
try:
    from governed_agent.governed_agent_core import Governor as RustGovernor
    HAS_RUST_CORE = True
except ImportError:
    RustGovernor = None
    HAS_RUST_CORE = False


class TestReviewBoundary(unittest.TestCase):
    """Verifies tripartite review boundary decision logic and Python/Rust parity."""

    def setUp(self):
        self.py_gov = PyGovernor(reward=0.02, loss=0.10, prior=0.50, max_stages=4)
        self.rust_gov = RustGovernor(reward=0.02, loss=0.10, prior=0.50, max_stages=4) if HAS_RUST_CORE else None

    def test_high_belief_selects_auto(self):
        """When defect risk (1 - b)*L < review_cost, AUTO strictly dominates REVIEW."""
        # b = 0.98:
        # u_auto = 0.98 * 0.02 - 0.02 * 0.10 = 0.0176
        # u_review = 0.98 * 0.02 - 0.005 = 0.0146
        # u_auto > u_review > 0 -> AUTO
        d_py = self.py_gov.evaluate_review_boundary(0.98, review_cost=0.005, shadow_price_lambda=0.0)
        self.assertEqual(d_py.action, "AUTO")
        self.assertAlmostEqual(d_py.u_auto, 0.0176, places=6)
        self.assertAlmostEqual(d_py.dominant_utility, 0.0176, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.98, review_cost=0.005, shadow_price_lambda=0.0)
            self.assertEqual(d_rust.action, "AUTO")
            self.assertAlmostEqual(d_rust.u_auto, 0.0176, places=6)
            self.assertAlmostEqual(d_rust.dominant_utility, 0.0176, places=6)

    def test_intermediate_belief_slack_queue_selects_review(self):
        """When queue is slack (lambda=0), human review saves error risk and dominates."""
        # b = 0.70 < p* (0.833):
        # u_auto = 0.70 * 0.02 - 0.30 * 0.10 = -0.016
        # u_review = 0.70 * 0.02 - 0.002 = 0.012 > 0 -> REVIEW
        d_py = self.py_gov.evaluate_review_boundary(0.70, review_cost=0.002, shadow_price_lambda=0.0)
        self.assertEqual(d_py.action, "REVIEW")
        self.assertAlmostEqual(d_py.u_review, 0.012, places=6)
        self.assertAlmostEqual(d_py.dominant_utility, 0.012, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.70, review_cost=0.002, shadow_price_lambda=0.0)
            self.assertEqual(d_rust.action, "REVIEW")
            self.assertAlmostEqual(d_rust.u_review, 0.012, places=6)
            self.assertAlmostEqual(d_rust.dominant_utility, 0.012, places=6)

    def test_congested_queue_shadow_price_induces_abstain(self):
        """When capacity shadow price lambda exceeds net review value, Governor abstains."""
        # b = 0.70, review_cost = 0.002
        # If queue capacity is constrained with shadow price lambda = 0.015:
        # u_review = 0.70 * 0.02 - 0.002 - 0.015 = -0.003 < 0
        # u_auto = -0.016 < 0
        # u_abstain = 0.0 -> ABSTAIN
        d_py = self.py_gov.evaluate_review_boundary(0.70, review_cost=0.002, shadow_price_lambda=0.015)
        self.assertEqual(d_py.action, "ABSTAIN")
        self.assertAlmostEqual(d_py.dominant_utility, 0.0, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.70, review_cost=0.002, shadow_price_lambda=0.015)
            self.assertEqual(d_rust.action, "ABSTAIN")
            self.assertAlmostEqual(d_rust.dominant_utility, 0.0, places=6)

    def test_low_belief_selects_abstain(self):
        """Very low belief candidates with negative expected review value abstain immediately."""
        d_py = self.py_gov.evaluate_review_boundary(0.10, review_cost=0.005, shadow_price_lambda=0.0)
        self.assertEqual(d_py.action, "ABSTAIN")
        self.assertAlmostEqual(d_py.dominant_utility, 0.0, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.10, review_cost=0.005, shadow_price_lambda=0.0)
            self.assertEqual(d_rust.action, "ABSTAIN")
            self.assertAlmostEqual(d_rust.dominant_utility, 0.0, places=6)

    def test_imperfect_review_accuracy_attenuation(self):
        """Review accuracy < 1.0 discounts the expected review payoff."""
        # b = 0.80, reward = 0.02, cost = 0.002, accuracy = 0.50
        # u_review = 0.80 * 0.02 * 0.50 - 0.002 = 0.008 - 0.002 = 0.006
        d_py = self.py_gov.evaluate_review_boundary(0.80, review_cost=0.002, review_accuracy=0.50)
        self.assertAlmostEqual(d_py.u_review, 0.006, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.80, review_cost=0.002, review_accuracy=0.50)
            self.assertAlmostEqual(d_rust.u_review, 0.006, places=6)

    def test_validation_errors(self):
        """Invalid inputs must raise ValueError in both Python and Rust."""
        for gov in [self.py_gov] + ([self.rust_gov] if self.rust_gov else []):
            with self.assertRaises(ValueError):
                gov.evaluate_review_boundary(-0.1)
            with self.assertRaises(ValueError):
                gov.evaluate_review_boundary(1.1)
            with self.assertRaises(ValueError):
                gov.evaluate_review_boundary(0.5, review_cost=-0.01)
            with self.assertRaises(ValueError):
                gov.evaluate_review_boundary(0.5, shadow_price_lambda=-0.01)
            with self.assertRaises(ValueError):
                gov.evaluate_review_boundary(0.5, review_accuracy=1.5)

    def test_to_dict_serialization(self):
        """Verify dictionary serialization for telemetry/reporting integration."""
        d_py = self.py_gov.evaluate_review_boundary(0.98, review_cost=0.005)
        py_dict = d_py.to_dict()
        self.assertEqual(py_dict["action"], "AUTO")
        self.assertAlmostEqual(py_dict["u_auto"], 0.0176, places=6)

        if self.rust_gov:
            d_rust = self.rust_gov.evaluate_review_boundary(0.98, review_cost=0.005)
            rust_dict = d_rust.to_dict()
            self.assertEqual(rust_dict["action"], "AUTO")
            self.assertAlmostEqual(rust_dict["u_auto"], 0.0176, places=6)


if __name__ == "__main__":
    unittest.main()
