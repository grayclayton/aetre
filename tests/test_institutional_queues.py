"""Unit and integration tests for Empirical Institutional Queue Benchmark (Gray, 2026)."""
import json
import math
import os
import unittest
from pathlib import Path

from evaluate_institutional_queues import (
    calculate_gaussian_voi,
    evaluate_dataset_queues,
    kingman_queue_delay,
    load_nih_data,
    load_openreview_data,
    load_uspto_data,
    simulate_shadow_price_escalation_curve,
)
from governed_agent.cli import main
from governed_agent.knapsack import CandidateSubmission, KnapsackController


class TestInstitutionalQueues(unittest.TestCase):

    def test_load_openreview_data(self):
        records = load_openreview_data()
        self.assertEqual(len(records), 4)
        for r in records:
            self.assertIn("candidate_id", r)
            self.assertTrue(0.0 <= r["quality_p"] <= 1.0)
            self.assertTrue(r["variance"] > 0.0)
            self.assertIn("is_ground_truth_high", r)

    def test_load_nih_data(self):
        records = load_nih_data()
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertIn("candidate_id", r)
            self.assertTrue(0.0 <= r["quality_p"] <= 1.0)
            self.assertTrue(r["variance"] > 0.0)

    def test_load_uspto_data(self):
        records = load_uspto_data()
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertIn("candidate_id", r)
            self.assertTrue(0.0 <= r["quality_p"] <= 1.0)

    def test_gaussian_voi_properties(self):
        # Far from boundary with low variance -> VOI is zero
        voi_far = calculate_gaussian_voi(mu_0=0.20, sigma_0_sq=0.01, tau=0.8333)
        self.assertEqual(voi_far, 0.0)

        # Near boundary with high variance -> VOI is strictly positive
        voi_boundary = calculate_gaussian_voi(mu_0=0.82, sigma_0_sq=0.10, tau=0.8333)
        self.assertGreater(voi_boundary, 0.0)

    def test_kingman_queue_delay_properties(self):
        # Low utilization -> small waiting time
        w_low = kingman_queue_delay(0.50)
        self.assertAlmostEqual(w_low, 1.0, places=2)

        # Severe utilization -> heavy delay
        w_high = kingman_queue_delay(0.95)
        self.assertAlmostEqual(w_high, 19.0, places=2)

        # Saturated utilization rho >= 1.0 -> infinite delay
        w_inf = kingman_queue_delay(1.0)
        self.assertTrue(math.isinf(w_inf))

    def test_openreview_benchmark_policies(self):
        records = load_openreview_data()
        res = evaluate_dataset_queues("openreview", records, capacity_K=2)
        self.assertEqual(res.total_candidates_N, 4)
        self.assertEqual(res.capacity_K, 2)
        self.assertEqual(res.ground_truth_high_value_H_N, 3)

        # Proposition 1 bound: R_N <= min(1, 2/3) = 66.7%
        self.assertAlmostEqual(res.prop1_theoretical_recall_ceiling, 2.0 / 3.0, places=3)

        # Knapsack must achieve non-negative welfare and strictly respect capacity
        kp = res.policies["pillar5_knapsack"]
        self.assertLessEqual(kp.admitted_count, 2)
        self.assertGreater(kp.total_welfare, 0.0)

        # Naive FIFO blindly admits first two, which include suboptimal items with negative welfare
        fifo = res.policies["naive_fifo"]
        self.assertLess(fifo.total_welfare, kp.total_welfare)

    def test_proposition_2_shadow_price_escalation(self):
        curve = simulate_shadow_price_escalation_curve(capacity_K=5, arrival_multipliers=[5, 25, 100, 500, 1000])
        # Verify arrival counts
        self.assertEqual(len(curve), 5)

        # Shadow price lambda_K must be non-decreasing as N scales
        prices = [pt["shadow_price_lambda_K"] for pt in curve]
        self.assertEqual(prices[0], 0.0)  # Slack when N=5, K=5
        self.assertGreater(prices[-1], 0.010)  # Surges when N=1000, K=5
        for i in range(len(prices) - 1):
            self.assertLessEqual(prices[i], prices[i + 1] + 1e-6)

        # Proposition 1 recall ceiling must decline towards 0 as N -> infinity
        ceilings = [pt["prop1_recall_ceiling"] for pt in curve]
        self.assertAlmostEqual(ceilings[0], 1.0, places=1)
        self.assertLess(ceilings[-1], 0.10)

    def test_cli_queue_dataset_integration(self):
        # Test OpenReview
        ret = main(["queue", "--dataset", "openreview", "--capacity", "2"])
        self.assertEqual(ret, 0)

        # Test NIH
        ret = main(["queue", "--dataset", "nih", "--capacity", "2"])
        self.assertEqual(ret, 0)

        # Test USPTO
        ret = main(["queue", "--dataset", "uspto", "--capacity", "1"])
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
