"""Cross-Language Parity Tests for Pillar I: Bayesian Bellman Governor.

Verifies bit-for-bit equivalence between the native Rust core (`governed_agent_core.Governor`)
and the pure-Python reference implementation (`governed_agent.governor.Governor`).
"""
import math
import unittest

try:
    from governed_agent.governed_agent_core import Governor as RustGovernor
    HAS_RUST_CORE = True
except (ImportError, OSError):
    RustGovernor = None
    HAS_RUST_CORE = False

from governed_agent.governor import Governor as PyGovernor


@unittest.skipUnless(HAS_RUST_CORE, "Native Rust core (governed_agent_core) unavailable")
class TestGovernorParity(unittest.TestCase):
    """Rigorous equivalence tests between Python and Rust Governor implementations."""

    def assert_float_almost_equal(self, a: float, b: float, tol: float = 1e-10, msg: str = ""):
        self.assertTrue(
            math.isclose(a, b, abs_tol=tol, rel_tol=tol),
            f"{msg}: {a} != {b} (diff={abs(a - b):.2e}, tol={tol:.2e})"
        )

    def test_default_initialization(self):
        py_gov = PyGovernor()
        rust_gov = RustGovernor()

        self.assert_float_almost_equal(py_gov.reward, rust_gov.reward)
        self.assert_float_almost_equal(py_gov.loss, rust_gov.loss)
        self.assert_float_almost_equal(py_gov.prior, rust_gov.prior)
        self.assertEqual(py_gov.max_stages, rust_gov.max_stages)
        self.assert_float_almost_equal(py_gov.p_star, rust_gov.p_star)
        self.assert_float_almost_equal(py_gov.defect_leakage, rust_gov.defect_leakage)
        self.assert_float_almost_equal(py_gov.conforming_pass_rate, rust_gov.conforming_pass_rate)
        self.assertEqual(len(py_gov.cost_schedule), len(rust_gov.cost_schedule))

    def test_parameter_sweeps_and_state_evaluations(self):
        test_configs = [
            # Standard config
            {"reward": 0.02, "loss": 0.10, "prior": 0.50, "max_stages": 4},
            # Symmetric stakes
            {"reward": 1.0, "loss": 1.0, "prior": 0.50, "max_stages": 3},
            # High stakes ratio (50x)
            {"reward": 0.01, "loss": 0.50, "prior": 0.70, "max_stages": 5},
            # Low prior belief
            {"reward": 0.05, "loss": 0.20, "prior": 0.15, "max_stages": 4},
            # High prior belief
            {"reward": 0.10, "loss": 0.10, "prior": 0.95, "max_stages": 4},
            # Custom cost schedule
            {
                "reward": 0.05,
                "loss": 0.25,
                "prior": 0.60,
                "max_stages": 3,
                "cost_schedule": [0.0005, 0.0010, 0.0020],
                "defect_leakage": 0.50,
                "conforming_pass_rate": 0.98,
            },
            # High leakage rate
            {
                "reward": 0.02,
                "loss": 0.10,
                "prior": 0.50,
                "max_stages": 5,
                "defect_leakage": 0.85,
                "conforming_pass_rate": 0.95,
            },
        ]

        for config in test_configs:
            with self.subTest(config=config):
                py_gov = PyGovernor(**config)
                rust_gov = RustGovernor(**config)

                # 1. p_star threshold check
                self.assert_float_almost_equal(
                    py_gov.p_star, rust_gov.p_star, msg="p_star mismatch"
                )

                # 2. Terminal utility sweep
                for b_int in range(101):
                    b = b_int / 100.0
                    self.assert_float_almost_equal(
                        py_gov.terminal_utility(b),
                        rust_gov.terminal_utility(b),
                        msg=f"terminal_utility({b}) mismatch"
                    )

                # 3. Posterior belief updates
                for b_int in [10, 30, 50, 70, 90]:
                    b = b_int / 100.0
                    for outcome in [True, False]:
                        py_post = py_gov.posterior_belief(b, outcome, 0)
                        rust_post = rust_gov.posterior_belief(b, outcome, 0)
                        self.assert_float_almost_equal(
                            py_post, rust_post, msg=f"posterior({b}, {outcome}) mismatch"
                        )

                # 4. Pass count lattice beliefs
                for k in range(py_gov.max_stages + 1):
                    py_bk = py_gov.belief_at_pass_count(k)
                    rust_bk = rust_gov.belief_at_pass_count(k)
                    self.assert_float_almost_equal(
                        py_bk, rust_bk, msg=f"belief_at_pass_count({k}) mismatch"
                    )

                # 5. Full state-space dynamic programming policy and VOI check
                for stage in range(py_gov.max_stages + 1):
                    for passes in range(stage + 1):
                        py_dec = py_gov.evaluate_state(stage, passes)
                        rust_dec = rust_gov.evaluate_state(stage, passes)

                        # Decisions must be identical
                        self.assertEqual(
                            py_dec.action, rust_dec.action,
                            f"Action mismatch at stage={stage}, passes={passes}"
                        )
                        self.assertEqual(py_dec.stage, rust_dec.stage)
                        self.assert_float_almost_equal(
                            py_dec.belief, rust_dec.belief,
                            msg=f"Belief mismatch at ({stage}, {passes})"
                        )
                        self.assert_float_almost_equal(
                            py_dec.expected_utility, rust_dec.expected_utility,
                            msg=f"Expected utility mismatch at ({stage}, {passes})"
                        )
                        self.assert_float_almost_equal(
                            py_dec.voi, rust_dec.voi,
                            msg=f"VOI mismatch at ({stage}, {passes})"
                        )

                        # Myopic paralysis predicate check
                        py_paralyzed = py_gov.is_myopic_paralyzed(stage, passes)
                        rust_paralyzed = rust_gov.is_myopic_paralyzed(stage, passes)
                        self.assertEqual(
                            py_paralyzed, rust_paralyzed,
                            f"Myopic paralysis mismatch at ({stage}, {passes})"
                        )

    def test_error_handling_parity(self):
        """Ensure both engines enforce exact domain constraints."""
        invalid_cases = [
            {"reward": -1.0, "loss": 1.0},
            {"reward": 1.0, "loss": -1.0},
            {"reward": 1.0, "loss": 1.0, "prior": 0.0},
            {"reward": 1.0, "loss": 1.0, "prior": 1.0},
            {"reward": 1.0, "loss": 1.0, "prior": 1.5},
            {"reward": 1.0, "loss": 1.0, "defect_leakage": 1.5},
            {"reward": 1.0, "loss": 1.0, "max_stages": 3, "cost_schedule": [0.01, 0.02]},
        ]

        for inv in invalid_cases:
            with self.subTest(inv=inv):
                with self.assertRaises(ValueError):
                    PyGovernor(**inv)
                with self.assertRaises(ValueError):
                    RustGovernor(**inv)


if __name__ == "__main__":
    unittest.main()
