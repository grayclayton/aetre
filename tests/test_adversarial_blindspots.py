"""Adversarial LLM Blind-Spot & Pathological Failure Stress Suite.

Tests how the Governed Agent (Pillars I-V and Road A) protects production systems
against subtle, deceptive, and pathological failure modes common in autonomous LLM agents:
1. Tautological / Vacuous Tests (Pillar IV Mutation Engine defense)
2. Stealth In-Place Argument Mutation (Pillar III Runtime Telemetry defense)
3. High-Entropy Erratic State Volatility (Pillar III Entropy penalty)
4. Syntax Poisoning & Malformed ASTs (Tier 0 Host-level zero-cost rejection)
5. Subtle Boundary Off-by-One Regressions (Tier 2 Host Mutation Oracles)
6. Saturated Review Spam Burst (Pillar V Knapsack shadow price shielding)
"""
from __future__ import annotations

import unittest
from governed_agent import (
    Governor,
    KnapsackController,
    CandidateSubmission,
    TieredVerificationGate,
    RuntimeTelemetry,
    MutationEngine,
    TieredPyramid,
)


class TestAdversarialBlindSpots(unittest.TestCase):
    """Stress tests against adversarial code generation and LLM blind spots."""

    def setUp(self):
        self.gov = Governor(reward=0.02, loss=0.10, prior=0.50, max_stages=4)
        self.gate = TieredVerificationGate()
        self.telemetry = RuntimeTelemetry()
        self.mutator = MutationEngine(qualification_threshold=0.80)

    def test_tautological_vacuous_test_attack(self):
        """Pillar IV: Vacuous or self-referential tests fail to kill mutants and are rejected."""
        # Clean reference implementation
        def ref_discount(price: float, rate: float) -> float:
            return price * (1.0 - rate)

        ref_code = """def ref_discount(price, rate):
    return price * (1.0 - rate)
"""
        # An LLM writes vacuous tests that pass even when logic is inverted (+ instead of -)
        vacuous_probes = [
            {"name": "vacuous_p1", "input": {"price": 0.0, "rate": 0.0}},
        ]

        mutants = self.mutator.generate_mutants(ref_code)
        self.assertGreater(len(mutants), 0)

        report = self.mutator.evaluate_mutation_score(
            probes=vacuous_probes,
            reference_fn=ref_discount,
            mutants=mutants,
        )

        # Mutants like price * (1.0 + rate) survive on price=0.0!
        self.assertFalse(
            report.is_qualified,
            f"Vacuous test suite should NOT qualify (Score: {report.mutation_score*100:.1f}%)"
        )
        self.assertLess(report.mutation_score, 0.80)
        self.assertGreater(report.survived_mutants, 0)

    def test_stealth_in_place_argument_mutation_attack(self):
        """Pillar III: Stealth in-place argument mutation is detected and severely penalizes belief."""
        # Sneaky candidate that mutates input dictionary in-place
        def sneaky_normalize(payload: dict) -> int:
            payload["_injected_side_effect"] = True  # Destructive in-place mutation!
            return len(payload)

        # Reference that is pure (no side effects)
        def pure_normalize(payload: dict) -> int:
            return len(payload)

        # Profile execution under telemetry
        self.telemetry.reset()
        test_input = {"user": "alice", "role": "admin"}
        res, err, lat, mutated = self.telemetry.profile_call(sneaky_normalize, {"payload": test_input})

        t_vec = self.telemetry.get_telemetry_vector()
        self.assertTrue(t_vec.input_mutated, "Telemetry must detect in-place argument mutation")

        # Telemetry calibration must severely degrade prior
        calibrated_prior = self.telemetry.calibrate_prior(t_vec, base_prior=0.50)
        self.assertAlmostEqual(calibrated_prior, 0.15, places=4)  # 0.50 * 0.30

        # With prior degraded to 0.15, even 3 passing probes cannot reach critical p* (83.33%)
        b = calibrated_prior
        for s in range(3):
            b = self.gov.posterior_belief(b, True, s)
        self.assertLess(b, self.gov.p_star, f"Degraded prior candidate must not breach p* ({b*100:.1f}% < 83.3%)")

    def test_high_entropy_state_volatility_penalty(self):
        """Pillar III: Erratic state representation entropy triggers a 30% volatility penalty."""
        # Simulated high-entropy state vector
        from governed_agent.telemetry import TelemetryVector
        high_entropy_vec = TelemetryVector(
            branch_coverage_pct=0.85,
            coverage_velocity=10.0,
            state_entropy=5.2,  # > 4.0 threshold indicates high erratic volatility
            latency_mean_ms=0.5,
            latency_jitter_ms=0.2,
            input_mutated=False,
            unique_states_observed=50,
        )

        calibrated = self.telemetry.calibrate_prior(high_entropy_vec, base_prior=0.50)
        self.assertAlmostEqual(calibrated, 0.35, places=4)  # 0.50 * 0.70 volatility penalty

    def test_syntax_poisoning_tier0_zero_cost_rejection(self):
        """Road A: Malformed AST / syntax errors short-circuit at Tier 0 in <1ms at $0 cost."""
        malformed_patch = """
def broken_syntax(x, y):
    if x > 0
        return y + 1
"""
        receipt = self.gate.verify_candidate(
            candidate_id="ATTACK-SYNTAX-01",
            candidate_code=malformed_patch,
            probes=[{"name": "p1", "input": {"x": 1, "y": 2}}],
        )

        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 0)
        self.assertTrue(receipt.short_circuited)
        self.assertFalse(receipt.docker_invoked)
        self.assertEqual(receipt.total_cost, 0.0)
        self.assertIn("Tier 0", receipt.diagnostics[0])
        self.assertLess(receipt.total_latency_sec, 0.01)

    def test_subtle_boundary_off_by_one_caught_at_tier2(self):
        """Road A: Candidate passing smoke fuzzing fails boundary oracle at Tier 2 without Docker."""
        # Candidate has off-by-one bug at boundary x = 0
        cand_code = "def is_positive(x):\n    return x > 0\n"
        ref_code = "def is_positive(x):\n    return x >= 0\n"

        cand_fn = lambda x: x > 0
        ref_fn = lambda x: x >= 0

        # Tier 1 probe passes on x = 5; Tier 2 boundary probe fails on x = 0
        probes = [
            {"name": "t1_smoke_fuzz", "tier": 1, "input": {"x": 5}},
            {"name": "t2_boundary_oracle", "tier": 2, "input": {"x": 0}},
        ]

        receipt = self.gate.verify_candidate(
            candidate_id="PR-OFF-BY-ONE",
            candidate_code=cand_code,
            candidate_fn=cand_fn,
            reference_fn=ref_fn,
            probes=probes,
        )

        self.assertFalse(receipt.admitted)
        self.assertEqual(receipt.terminal_tier, 2)
        self.assertTrue(receipt.short_circuited)
        self.assertFalse(receipt.docker_invoked)
        self.assertEqual(receipt.final_belief, 0.0)
        self.assertLessEqual(receipt.total_cost, 0.001)

    def test_saturated_review_spam_shadow_price_shielding(self):
        """Pillar V: High-volume review bursts spike lambda_K, flipping review to abstain."""
        knapsack = KnapsackController(default_capacity=2, review_cost_k=1)

        # Candidates where excluded candidate is intrinsically acceptable (b=0.88 > p*=0.833)
        # creating a strictly positive shadow price lambda_K > 0
        candidates = [
            CandidateSubmission(candidate_id="PR-HIGH-1", task_id="auth", posterior_belief=0.96),
            CandidateSubmission(candidate_id="PR-HIGH-2", task_id="billing", posterior_belief=0.94),
            CandidateSubmission(candidate_id="PR-QUALIFIED-3", task_id="cache", posterior_belief=0.88),
            CandidateSubmission(candidate_id="PR-MED-4", task_id="t4", posterior_belief=0.70),
        ]

        report = knapsack.admit_batch(candidates, capacity_K=2, review_cost_k=1)
        self.assertEqual(report.total_admitted, 2)
        self.assertEqual(report.total_rejected, 2)
        self.assertGreater(report.shadow_price_lambda, 0.0)

        # Marginal candidate (b = 0.70, review_cost = 0.009)
        # Under slack queue (lambda = 0.0): u_review = 0.70 * 0.02 - 0.009 = +0.0050 > 0 -> REVIEW
        slack_decision = self.gov.evaluate_review_boundary(
            belief=0.70, review_cost=0.009, shadow_price_lambda=0.0
        )
        self.assertEqual(slack_decision.action, "REVIEW")

        # Under congested queue (lambda = shadow_price_lambda):
        # u_review = 0.70 * 0.02 - 0.009 - lambda_K < 0 -> ABSTAIN!
        congested_decision = self.gov.evaluate_review_boundary(
            belief=0.70, review_cost=0.009, shadow_price_lambda=report.shadow_price_lambda
        )
        self.assertEqual(congested_decision.action, "ABSTAIN")
        self.assertLess(congested_decision.u_review, 0.0)


if __name__ == "__main__":
    unittest.main()
