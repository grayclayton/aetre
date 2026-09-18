"""Pillar III: Dynamic Runtime Execution Telemetry.

Monitors dynamic execution traces during test execution:
- Branch coverage velocity (Delta Cov / Delta t)
- State-transition Shannon entropy H(S)
- Argument mutation churn & execution jitter

Demonstrates that dynamic runtime signals break the static AST complexity blindness.
"""
from __future__ import annotations

import collections
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class TelemetryVector:
    branch_coverage_pct: float
    coverage_velocity: float
    state_entropy: float
    latency_mean_ms: float
    latency_jitter_ms: float
    input_mutated: bool
    unique_states_observed: int


class RuntimeTelemetry:
    """Monitors dynamic runtime traces during function execution."""

    def __init__(self):
        self.executed_lines: Set[Tuple[str, int]] = set()
        self.observed_states: List[Any] = []
        self.latencies: List[float] = []
        self.mutation_events: int = 0
        self._trace_active: bool = False

    def reset(self) -> None:
        """Reset internal telemetry accumulators."""
        self.executed_lines.clear()
        self.observed_states.clear()
        self.latencies.clear()
        self.mutation_events = 0

    def compute_shannon_entropy(self) -> float:
        """Calculate Shannon entropy over observed discrete state traces: H(S) = -sum p(s) log2 p(s)."""
        if not self.observed_states:
            return 0.0
        # Convert state representations to hashable strings
        counts = collections.Counter(str(s) for s in self.observed_states)
        total = len(self.observed_states)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    def profile_call(
        self,
        func: Callable[..., Any],
        kwargs: Dict[str, Any],
        expected_total_lines: int = 20,
    ) -> Tuple[Any, Optional[Exception], float, bool]:
        """Execute a function under trace profiling, recording lines, states, and mutations."""
        import copy
        kwargs_copy = copy.deepcopy(kwargs)
        prior_coverage_count = len(self.executed_lines)

        captured_lines: Set[Tuple[str, int]] = set()
        captured_states: List[Any] = []

        def trace_handler(frame, event, arg):
            if event == "line":
                code = frame.f_code
                filename = code.co_filename
                lineno = frame.f_lineno
                # Restrict to function under test
                if code.co_name == func.__name__ or "natural_benchmark" in filename or "ref_" in code.co_name:
                    captured_lines.add((code.co_name, lineno))
                    # Record snapshot of local variables for entropy
                    locals_snapshot = tuple(sorted((k, repr(v)[:30]) for k, v in frame.f_locals.items()))
                    captured_states.append(locals_snapshot)
            return trace_handler

        t0 = time.perf_counter()
        old_trace = sys.gettrace()
        res = None
        err = None
        sys.settrace(trace_handler)
        try:
            res = func(**kwargs)
        except Exception as ex:
            err = ex
        finally:
            sys.settrace(old_trace)
            latency = time.perf_counter() - t0

        self.latencies.append(latency)
        self.executed_lines.update(captured_lines)
        self.observed_states.extend(captured_states)

        # Check for input mutation side-effects
        mutated = (kwargs != kwargs_copy)
        if mutated:
            self.mutation_events += 1

        return res, err, latency, mutated

    def get_telemetry_vector(self, expected_total_lines: int = 20) -> TelemetryVector:
        """Summarize current execution profile into a TelemetryVector."""
        unique_lines = len(self.executed_lines)
        cov_pct = min(1.0, unique_lines / max(1, expected_total_lines))
        entropy = self.compute_shannon_entropy()

        n_latencies = len(self.latencies)
        mean_lat = (sum(self.latencies) / n_latencies * 1000.0) if n_latencies > 0 else 0.0
        if n_latencies > 1:
            variance = sum((l * 1000.0 - mean_lat) ** 2 for l in self.latencies) / (n_latencies - 1)
            jitter = math.sqrt(variance)
        else:
            jitter = 0.0

        # Coverage velocity: executed lines per probe call
        cov_velocity = (unique_lines / max(1, n_latencies))

        return TelemetryVector(
            branch_coverage_pct=cov_pct,
            coverage_velocity=cov_velocity,
            state_entropy=entropy,
            latency_mean_ms=mean_lat,
            latency_jitter_ms=jitter,
            input_mutated=self.mutation_events > 0,
            unique_states_observed=len(self.observed_states),
        )

    def calibrate_prior(self, vector: TelemetryVector, base_prior: float = 0.50) -> float:
        """Adjust candidate prior belief based on dynamic execution telemetry.

        - Anomalous state entropy shifts or argument mutations degrade prior.
        - High branch coverage velocity and deterministic low jitter boost prior.
        """
        prior = base_prior
        # Penalty for argument mutation side-effects
        if vector.input_mutated:
            prior *= 0.30

        # Penalty for high state volatility / unbounded entropy
        if vector.state_entropy > 4.0:
            prior *= 0.70
        elif vector.state_entropy < 1.5 and vector.branch_coverage_pct > 0.60:
            # Clean, deterministic execution with high coverage
            prior = 1.0 - (1.0 - prior) * 0.60

        return max(0.01, min(0.99, prior))
