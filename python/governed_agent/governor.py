"""Pillar I: Decoupled Economic Governor (Python Reference & Wrapper).

Implements the Bayesian Bellman supervisory controller that regulates
verification effort, evaluates multi-step Value of Information (VOI),
and enforces stopping and admission rules under asymmetric loss stakes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Decision:
    action: str  # 'HALT_AND_COMMIT', 'HALT_AND_REJECT', 'CONTINUE'
    stage: int
    belief: float
    expected_utility: float
    probe_idx: Optional[int] = None
    voi: float = 0.0


@dataclass
class ReviewBoundaryDecision:
    """Outcome of Section 4.1 tripartite review boundary evaluation.

    Formalized in Gray (2026d) The Admission Frontier:
        E[U(auto, theta) | z_t] >= max(E[U(review, theta) | z_t], E[U(abstain, theta) | z_t])
    """
    action: str  # "AUTO", "REVIEW", or "ABSTAIN"
    u_auto: float
    u_review: float
    u_abstain: float
    belief: float
    shadow_price_lambda: float
    dominant_utility: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "u_auto": self.u_auto,
            "u_review": self.u_review,
            "u_abstain": self.u_abstain,
            "belief": self.belief,
            "shadow_price_lambda": self.shadow_price_lambda,
            "dominant_utility": self.dominant_utility,
        }


class Governor:
    """Non-autoregressive Bayesian supervisor executing dynamic stopping."""

    def __init__(
        self,
        reward: float = 0.02,
        loss: float = 0.10,
        prior: float = 0.50,
        max_stages: int = 4,
        cost_schedule: Optional[List[float]] = None,
        defect_leakage: float = 0.6125,
        conforming_pass_rate: float = 1.0,
    ):
        if reward <= 0 or loss <= 0:
            raise ValueError("Reward and loss must be strictly positive")
        if not (0.0 < prior < 1.0):
            raise ValueError("Prior belief must be in (0, 1)")
        if not (0.0 <= defect_leakage <= 1.0):
            raise ValueError("Defect leakage q must be in [0, 1]")

        self.reward = reward
        self.loss = loss
        self.prior = prior
        self.max_stages = max_stages
        self.p_star = loss / (reward + loss)
        self.defect_leakage = defect_leakage
        self.conforming_pass_rate = conforming_pass_rate

        if cost_schedule is None:
            self.cost_schedule = [1e-6] * max_stages
        else:
            if len(cost_schedule) != max_stages:
                raise ValueError(f"Cost schedule length ({len(cost_schedule)}) must equal max_stages ({max_stages})")
            self.cost_schedule = [float(c) for c in cost_schedule]

        self._dp_cache: Dict[Tuple[int, int], float] = {}
        self._policy_cache: Dict[Tuple[int, int], str] = {}
        self._solve_dp()

    def terminal_utility(self, belief: float) -> float:
        accept_utility = belief * self.reward - (1.0 - belief) * self.loss
        return max(0.0, accept_utility)

    def posterior_belief(self, prior_b: float, outcome_pass: bool, probe_idx: int = 0) -> float:
        p_pass_given_conf = self.conforming_pass_rate
        p_pass_given_def = self.defect_leakage

        if outcome_pass:
            num = prior_b * p_pass_given_conf
            den = num + (1.0 - prior_b) * p_pass_given_def
        else:
            num = prior_b * (1.0 - p_pass_given_conf)
            den = num + (1.0 - prior_b) * (1.0 - p_pass_given_def)

        if den <= 0.0:
            return 0.0 if not outcome_pass else 1.0
        return max(0.0, min(1.0, num / den))

    def belief_at_pass_count(self, pass_count: int) -> float:
        q_eff = self.defect_leakage ** pass_count
        p_conf = self.conforming_pass_rate ** pass_count
        num = self.prior * p_conf
        den = num + (1.0 - self.prior) * q_eff
        if den <= 0:
            return 0.0
        return num / den

    def _solve_dp(self) -> None:
        self._dp_cache.clear()
        self._policy_cache.clear()

        for k in range(self.max_stages + 1):
            b_k = self.belief_at_pass_count(k)
            u_term = self.terminal_utility(b_k)
            self._dp_cache[(self.max_stages, k)] = u_term
            self._policy_cache[(self.max_stages, k)] = "ACCEPT" if b_k >= self.p_star else "REJECT"

        for t in range(self.max_stages - 1, -1, -1):
            cost = self.cost_schedule[t]
            for k in range(t + 1):
                b = self.belief_at_pass_count(k)
                u_stop = self.terminal_utility(b)

                p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage
                p_fail = 1.0 - p_pass

                v_pass = self._dp_cache[(t + 1, k + 1)]
                v_fail = 0.0

                v_continue = -cost + (p_pass * v_pass + p_fail * v_fail)

                if v_continue > u_stop + 1e-12:
                    self._dp_cache[(t, k)] = v_continue
                    self._policy_cache[(t, k)] = "PROBE"
                else:
                    self._dp_cache[(t, k)] = u_stop
                    self._policy_cache[(t, k)] = "ACCEPT" if b >= self.p_star else "REJECT"

    def evaluate_state(self, stage: int, consecutive_passes: int) -> Decision:
        if stage > self.max_stages:
            stage = self.max_stages
        if consecutive_passes > stage:
            consecutive_passes = stage

        b = self.belief_at_pass_count(consecutive_passes)
        u_stop = self.terminal_utility(b)
        expected_v = self._dp_cache.get((stage, consecutive_passes), u_stop)
        action_name = self._policy_cache.get((stage, consecutive_passes), "REJECT")

        voi = 0.0
        if stage < self.max_stages:
            p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage
            p_fail = 1.0 - p_pass
            v_pass = self._dp_cache[(stage + 1, consecutive_passes + 1)]
            v_fail = 0.0
            e_next = p_pass * v_pass + p_fail * v_fail
            voi = max(0.0, e_next - u_stop)

        if action_name == "PROBE":
            return Decision(
                action="CONTINUE",
                stage=stage,
                belief=b,
                expected_utility=expected_v,
                probe_idx=stage,
                voi=voi,
            )
        elif action_name == "ACCEPT":
            return Decision(
                action="HALT_AND_COMMIT",
                stage=stage,
                belief=b,
                expected_utility=expected_v,
                voi=0.0,
            )
        else:
            return Decision(
                action="HALT_AND_REJECT",
                stage=stage,
                belief=b,
                expected_utility=0.0,
                voi=0.0,
            )

    def is_myopic_paralyzed(self, stage: int = 0, consecutive_passes: int = 0) -> bool:
        b = self.belief_at_pass_count(consecutive_passes)
        u_stop = self.terminal_utility(b)
        if stage >= self.max_stages:
            return False

        cost = self.cost_schedule[stage]
        p_pass = b * self.conforming_pass_rate + (1.0 - b) * self.defect_leakage
        b_next = self.posterior_belief(b, True, stage)
        u_next_term = self.terminal_utility(b_next)
        myopic_continue_v = -cost + p_pass * u_next_term

        myopic_wants_stop = myopic_continue_v <= u_stop
        dp_wants_continue = self._policy_cache.get((stage, consecutive_passes)) == "PROBE"
        return myopic_wants_stop and dp_wants_continue

    def evaluate_review_boundary(
        self,
        belief: float,
        review_cost: float = 0.005,
        shadow_price_lambda: float = 0.0,
        review_accuracy: float = 1.0,
    ) -> ReviewBoundaryDecision:
        """Evaluates the tripartite review boundary from Section 4.1 (Gray 2026d).

        Compares:
        1. E[U(auto, theta) | z_t] = max(0, b * R - (1 - b) * L)
        2. E[U(review, theta) | z_t] = (b * R * acc) - review_cost - lambda_K
        3. E[U(abstain, theta) | z_t] = 0.0

        Routes to AUTO if u_auto >= max(u_review, u_abstain),
        REVIEW if u_review > u_abstain,
        otherwise ABSTAIN.
        """
        if not (0.0 <= belief <= 1.0):
            raise ValueError("Belief must be in [0, 1]")
        if review_cost < 0.0:
            raise ValueError("Review cost must be non-negative")
        if shadow_price_lambda < 0.0:
            raise ValueError("Shadow price lambda must be non-negative")
        if not (0.0 <= review_accuracy <= 1.0):
            raise ValueError("Review accuracy must be in [0, 1]")

        u_auto = belief * self.reward - (1.0 - belief) * self.loss
        u_review = (belief * self.reward * review_accuracy) - review_cost - shadow_price_lambda
        u_abstain = 0.0

        if u_auto >= max(u_review, u_abstain):
            action = "AUTO"
            dom = u_auto
        elif u_review > u_abstain:
            action = "REVIEW"
            dom = u_review
        else:
            action = "ABSTAIN"
            dom = u_abstain

        return ReviewBoundaryDecision(
            action=action,
            u_auto=u_auto,
            u_review=u_review,
            u_abstain=u_abstain,
            belief=belief,
            shadow_price_lambda=shadow_price_lambda,
            dominant_utility=dom,
        )
