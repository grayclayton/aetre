"""Unit tests for Proposition 1 Defect Bound & Queue Congestion Dynamics.

Runs a fast Monte Carlo verification (1,500 candidates) asserting:
1. Proposition 1 Bound: Defect rate in AUTO is bounded by 1 - p* = R / (R + L).
2. Capacity Shadow Price Dynamics: Congestion flips marginal candidate from REVIEW to ABSTAIN.
3. Road A Efficiency: Host-level pre-checks bypass >40% of container executions.
"""
from __future__ import annotations

import unittest
from benchmarks.simulate_bounds_and_congestion import run_monte_carlo_simulation


class TestMonteCarloBounds(unittest.TestCase):
    """Verifies that the theoretical safety bounds and queue dynamics hold under simulation."""

    def test_monte_carlo_proposition_1_and_queue_dynamics(self):
        # Run fast simulation with 1,500 candidates
        results = run_monte_carlo_simulation(n_samples=1500, seed=123, verbose=False)

        # 1. Verify Proposition 1 Bound across all calibrated regimes
        p1_data = results.proposition_1_bound
        for regime_name, data in p1_data.items():
            cal = data["calibrated_governor"]
            self.assertTrue(
                cal["bound_satisfied"],
                f"Proposition 1 bound violated in {regime_name}: defect rate {cal['defect_rate_pct']}% > 16.67%"
            )
            self.assertGreaterEqual(cal["safety_margin_pct"], 0.0)

        # 2. Verify Queue Congestion Dynamic Shielding
        series = results.queue_congestion["series"]
        # Find low load (rho <= 1.0) and high load (rho >= 3.0)
        low_load = [s for s in series if s["traffic_intensity_rho"] <= 1.0][-1]
        high_load = [s for s in series if s["traffic_intensity_rho"] >= 3.0][0]

        # Under low load, marginal candidate (b=0.70) must be routed to REVIEW
        self.assertEqual(low_load["marginal_candidate_decision"], "REVIEW")
        self.assertGreater(low_load["marginal_review_net_utility"], 0.0)

        # Under high load, shadow price lambda must spike and flip marginal candidate to ABSTAIN
        self.assertEqual(high_load["marginal_candidate_decision"], "ABSTAIN")
        self.assertLess(high_load["marginal_review_net_utility"], 0.0)
        self.assertGreater(high_load["capacity_shadow_price_lambda"], 0.010)

        # 3. Verify Road A Economics
        econ = results.economics
        self.assertGreater(econ["cost_savings_pct"], 90.0)
        self.assertGreater(econ["container_avoidance_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
