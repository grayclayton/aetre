"""Pillar V: Capacity-Aware Knapsack Admission (Python Reference & Wrapper).

Regulates downstream review and staging queues according to the
macroeconomic and absorption constraints established in Gray (2026a, b):
- Enforces optimal selection depth m* = min(n, floor(K/k)) under uniform workloads
- Implements density ranking rho_i = E[U_i] / k_i and queuing-theoretic c-mu scheduling
- Tracks fractional review capacity, total admitted effort, and dual shadow price Delta_K W
- Prevents candidate pull request flooding under bounded review bandwidth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


@dataclass
class CandidateSubmission:
    candidate_id: str
    task_id: str
    posterior_belief: float
    reward: float = 0.02
    loss: float = 0.10
    raw_output: Any = None
    review_cost: Optional[float] = None

    def __post_init__(self) -> None:
        if self.review_cost is not None and self.review_cost <= 0.0:
            raise ValueError(f"Review cost must be positive, got {self.review_cost}")

    @property
    def expected_utility(self) -> float:
        return self.posterior_belief * self.reward - (1.0 - self.posterior_belief) * self.loss

    @property
    def p_star(self) -> float:
        return self.loss / (self.reward + self.loss)

    @property
    def is_intrinsically_acceptable(self) -> bool:
        return self.posterior_belief >= self.p_star

    @property
    def effective_review_cost(self) -> float:
        """Effective review cost when evaluated in isolation (defaults to 1.0)."""
        return self.review_cost if self.review_cost is not None else 1.0

    @property
    def density(self) -> float:
        """Value density rho_i = E[U_i] / k_i (the c-mu index)."""
        return self.expected_utility / self.effective_review_cost

    @property
    def c_mu_index(self) -> float:
        """Queuing-theoretic c-mu scheduling index."""
        return self.density

    @property
    def review_cost_k(self) -> Optional[float]:
        """Alias for review_cost for backward compatibility."""
        return self.review_cost


@dataclass
class KnapsackAdmissionReport:
    capacity_K: float
    review_cost_k: float
    optimal_selection_depth: int
    total_candidates: int
    total_admitted: int
    total_rejected: int
    total_welfare: float
    shadow_price_lambda: float
    admitted: List[CandidateSubmission] = field(default_factory=list)
    rejected: List[CandidateSubmission] = field(default_factory=list)
    total_admitted_cost: float = 0.0
    remaining_capacity: float = 0.0


class KnapsackController:
    """Regulates downstream queue admission under finite absorption capacity."""

    def __init__(self, default_capacity: float = 10, review_cost_k: float = 1):
        if default_capacity <= 0 or review_cost_k <= 0:
            raise ValueError("Capacity and review cost must be positive")
        self.capacity_K = default_capacity
        self.review_cost_k = review_cost_k

    def admit_batch(
        self,
        candidates: List[CandidateSubmission],
        capacity_K: Optional[float] = None,
        review_cost_k: Optional[float] = None,
    ) -> KnapsackAdmissionReport:
        K = float(capacity_K if capacity_K is not None else self.capacity_K)
        k = float(review_cost_k if review_cost_k is not None else self.review_cost_k)
        if K <= 0 or k <= 0:
            raise ValueError("Capacity and review cost must be positive")

        n = len(candidates)

        # Check if workload is strictly homogeneous with respect to k
        # Candidates without explicit review_cost inherit the controller's review_cost_k
        is_homogeneous = all(
            c.review_cost is None or abs(c.review_cost - k) < 1e-9
            for c in candidates
        )

        def get_effective_cost(c: CandidateSubmission) -> float:
            return c.review_cost if c.review_cost is not None else k

        def get_density(c: CandidateSubmission) -> float:
            cost = get_effective_cost(c)
            return c.expected_utility / cost if cost > 0 else c.expected_utility

        # Rank candidates descending by value density (c-mu index = E[U] / cost)
        # breaking ties by expected utility
        ranked = sorted(
            candidates,
            key=lambda c: (get_density(c), c.expected_utility),
            reverse=True,
        )

        admitted: List[CandidateSubmission] = []
        rejected: List[CandidateSubmission] = []
        total_admitted_cost = 0.0

        for cand in ranked:
            cost = get_effective_cost(cand)
            if cand.is_intrinsically_acceptable and (total_admitted_cost + cost <= K + 1e-9):
                total_admitted_cost += cost
                admitted.append(cand)
            else:
                rejected.append(cand)

        total_welfare = sum(c.expected_utility for c in admitted)
        remaining_capacity = max(0.0, K - total_admitted_cost)

        # Shadow price Delta_K W
        if is_homogeneous and K.is_integer() and k.is_integer():
            K_int = int(K)
            k_int = int(k)
            m_star = min(n, K_int // k_int)
            next_depth = min(n, (K_int + 1) // k_int)
            if next_depth == m_star or len(admitted) < m_star:
                shadow_price = 0.0
            elif m_star < n and ranked[m_star].is_intrinsically_acceptable:
                shadow_price = ranked[m_star].expected_utility
            else:
                shadow_price = 0.0
            optimal_depth = m_star
        else:
            admitted_ids = {c.candidate_id for c in admitted}
            unadmitted_acceptable = [
                c for c in ranked
                if c.is_intrinsically_acceptable and c.candidate_id not in admitted_ids
            ]
            if unadmitted_acceptable:
                shadow_price = get_density(unadmitted_acceptable[0])
            else:
                shadow_price = 0.0
            optimal_depth = len(admitted)

        return KnapsackAdmissionReport(
            capacity_K=K,
            review_cost_k=k,
            optimal_selection_depth=optimal_depth,
            total_candidates=n,
            total_admitted=len(admitted),
            total_rejected=len(rejected),
            total_welfare=total_welfare,
            shadow_price_lambda=shadow_price,
            admitted=admitted,
            rejected=rejected,
            total_admitted_cost=total_admitted_cost,
            remaining_capacity=remaining_capacity,
        )
