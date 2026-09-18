"""Performance Benchmark: Native Rust Core vs. Pure Python Implementation.

Measures latency, throughput, and speedup multiplier for:
1. Pillar I: Bayesian Bellman Governor (Dynamic Programming & State Evaluation)
2. Pillar V: Capacity-Aware Knapsack Queue Admission & Shadow Price Computation
"""
import time
import unittest
import random

try:
    from governed_agent.governed_agent_core import (
        Governor as RustGovernor,
        KnapsackController as RustKnapsack,
        CandidateSubmission as RustCandidate,
    )
    HAS_RUST_CORE = True
except (ImportError, OSError):
    RustGovernor = None
    RustKnapsack = None
    RustCandidate = None
    HAS_RUST_CORE = False

from governed_agent.governor import Governor as PyGovernor
from governed_agent.knapsack import (
    KnapsackController as PyKnapsack,
    CandidateSubmission as PyCandidate,
)


@unittest.skipUnless(HAS_RUST_CORE, "Native Rust core (governed_agent_core) unavailable")
class TestPerformanceBenchmark(unittest.TestCase):
    """Measures execution speedup of native Rust core over pure Python reference."""

    def test_governor_dp_and_evaluation_speedup(self):
        iterations = 50_000
        print(f"\n[BENCHMARK] Running Governor Evaluation ({iterations:,} iterations)...")

        # 1. Benchmark Pure Python
        py_gov = PyGovernor(reward=0.02, loss=0.10, prior=0.50, max_stages=6)
        start_py = time.perf_counter()
        for i in range(iterations):
            stage = i % 7
            passes = (i // 7) % (stage + 1)
            _ = py_gov.evaluate_state(stage, passes)
        duration_py = time.perf_counter() - start_py

        # 2. Benchmark Native Rust Core
        rust_gov = RustGovernor(reward=0.02, loss=0.10, prior=0.50, max_stages=6)
        start_rust = time.perf_counter()
        for i in range(iterations):
            stage = i % 7
            passes = (i // 7) % (stage + 1)
            _ = rust_gov.evaluate_state(stage, passes)
        duration_rust = time.perf_counter() - start_rust

        speedup = duration_py / duration_rust if duration_rust > 0 else float("inf")
        throughput_rust = iterations / duration_rust if duration_rust > 0 else 0
        throughput_py = iterations / duration_py if duration_py > 0 else 0

        print(f"  Python Time:     {duration_py * 1000:.2f} ms ({throughput_py:,.0f} ops/sec)")
        print(f"  Rust Time:       {duration_rust * 1000:.2f} ms ({throughput_rust:,.0f} ops/sec)")
        print(f"  Rust Speedup:    {speedup:.2f}x faster")

        self.assertGreater(speedup, 1.2, "Native Rust core should be faster than pure Python")

    def test_knapsack_admission_speedup(self):
        batches = 2_000
        cands_per_batch = 50
        print(f"\n[BENCHMARK] Running Knapsack Admission ({batches:,} batches of {cands_per_batch} candidates)...")

        rng = random.Random(1337)
        cands_raw = [
            (f"cand_{j}", rng.uniform(0.1, 0.99), 0.02, 0.10)
            for j in range(cands_per_batch)
        ]

        py_cands = [
            PyCandidate(candidate_id=cid, task_id="t1", posterior_belief=p, reward=r, loss=l)
            for cid, p, r, l in cands_raw
        ]
        rust_cands = [
            RustCandidate(candidate_id=cid, task_id="t1", posterior_belief=p, reward=r, loss=l)
            for cid, p, r, l in cands_raw
        ]

        py_ks = PyKnapsack(default_capacity=20, review_cost_k=2)
        rust_ks = RustKnapsack(default_capacity=20, review_cost_k=2)

        # 1. Benchmark Pure Python
        start_py = time.perf_counter()
        for _ in range(batches):
            _ = py_ks.admit_batch(py_cands)
        duration_py = time.perf_counter() - start_py

        # 2. Benchmark Native Rust Core
        start_rust = time.perf_counter()
        for _ in range(batches):
            _ = rust_ks.admit_batch(rust_cands)
        duration_rust = time.perf_counter() - start_rust

        speedup = duration_py / duration_rust if duration_rust > 0 else float("inf")
        throughput_rust = (batches * cands_per_batch) / duration_rust if duration_rust > 0 else 0
        throughput_py = (batches * cands_per_batch) / duration_py if duration_py > 0 else 0

        print(f"  Python Time:     {duration_py * 1000:.2f} ms ({throughput_py:,.0f} cands/sec)")
        print(f"  Rust Time:       {duration_rust * 1000:.2f} ms ({throughput_rust:,.0f} cands/sec)")
        print(f"  Rust Speedup:    {speedup:.2f}x faster")

        self.assertGreater(speedup, 1.2, "Native Rust core should be faster than pure Python")


if __name__ == "__main__":
    unittest.main()
