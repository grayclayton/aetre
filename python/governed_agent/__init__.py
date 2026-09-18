"""The Governed Agent: High-Performance Decision-Theoretic Runtime.

Provides sub-microsecond Bayesian Bellman dynamic programming (Pillar I)
and capacity-aware Knapsack queue admission (Pillar V) powered by a native Rust core.
"""
from __future__ import annotations

__version__ = "0.2.0"

HAS_RUST_CORE: bool = False

try:
    from .governed_agent_core import (
        Governor,
        Decision,
        ReviewBoundaryDecision,
        KnapsackController,
        CandidateSubmission,
        KnapsackAdmissionReport,
    )
    HAS_RUST_CORE = True
except ImportError:
    # Graceful fallback to pure-Python implementation
    from .governor import Governor, Decision, ReviewBoundaryDecision
    from .knapsack import KnapsackController, CandidateSubmission, KnapsackAdmissionReport
    HAS_RUST_CORE = False

from .pyramid import TieredPyramid, ProbeOutcome, TieredExecutionReport
from .telemetry import RuntimeTelemetry, TelemetryVector
from .isolation import ProcessIsolatedRunner, ExecutionStatus, IsolatedExecutionResult
from .mutator import MutationEngine, Mutant, MutationReport
from .gate import TieredVerificationGate, GateReceipt, GateBatchReport

__all__ = [
    "Governor",
    "Decision",
    "ReviewBoundaryDecision",
    "KnapsackController",
    "CandidateSubmission",
    "KnapsackAdmissionReport",
    "TieredPyramid",
    "ProbeOutcome",
    "TieredExecutionReport",
    "TieredVerificationGate",
    "GateReceipt",
    "GateBatchReport",
    "RuntimeTelemetry",
    "TelemetryVector",
    "ProcessIsolatedRunner",
    "ExecutionStatus",
    "IsolatedExecutionResult",
    "MutationEngine",
    "Mutant",
    "HAS_RUST_CORE",
    "__version__",
]
