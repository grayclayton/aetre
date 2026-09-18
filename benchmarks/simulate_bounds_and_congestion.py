"""Large-Scale Monte Carlo Simulation: Proposition 1 Defect Bound & Queue Congestion.

Formalizes and validates the theoretical results from:
- Paper 1 (Gray 2026a) The Innovation-Absorption Gap: Heavy-traffic delay and absorption saturation.
- Paper 4 (Gray 2026d) The Admission Frontier:
    1. Proposition 1 Defect Bound: False admission rate into AUTO <= 1 - p* = R / (R + L) = 16.67%.
    2. Capacity Shadow Price Dynamics: lambda_K = Delta_K W adjusts the Section 4.1 review boundary.
    3. Road A Economic Reduction: Host pre-checks bypass >70% of Docker container executions.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Use native Rust core if available, otherwise pure-Python fallback
from governed_agent import Governor, KnapsackController, CandidateSubmission, HAS_RUST_CORE


@dataclass
class SimulationResults:
    total_candidates: int
    duration_sec: float
    throughput_per_sec: float
    proposition_1_bound: Dict[str, Any]
    queue_congestion: Dict[str, Any]
    economics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "duration_sec": round(self.duration_sec, 4),
            "throughput_candidates_per_sec": round(self.throughput_per_sec, 1),
            "rust_native_core_active": HAS_RUST_CORE,
            "proposition_1_bound": self.proposition_1_bound,
            "queue_congestion": self.queue_congestion,
            "economics": self.economics,
        }


def run_monte_carlo_simulation(
    n_samples: int = 10000,
    reward: float = 0.02,
    loss: float = 0.10,
    prior: float = 0.50,
    defect_leakage: float = 0.5875,
    max_stages: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> SimulationResults:
    random.seed(seed)
    start_time = time.perf_counter()

    gov = Governor(
        reward=reward,
        loss=loss,
        prior=prior,
        max_stages=max_stages,
        cost_schedule=[0.0001, 0.0005, 0.0005, 0.0020],
        defect_leakage=defect_leakage,
    )
    p_star = gov.p_star
    theoretical_max_defect_rate = 1.0 - p_star  # R / (R + L) = 16.67%

    if verbose:
        print("=" * 80)
        print("  MONTE CARLO SIMULATION: PROPOSITION 1 DEFECT BOUND & QUEUE CONGESTION")
        print("=" * 80)
        print(f"Total Candidates:         {n_samples:,}")
        print(f"Engine:                   {'Rust Native Core' if HAS_RUST_CORE else 'Pure-Python Fallback'}")
        print(f"Stakes:                   Reward R = ${reward:.4f} | Loss L = ${loss:.4f} (Stakes Ratio {loss/reward:.1f}x)")
        print(f"Critical Threshold p*:    {p_star * 100:.2f}% | Proposition 1 Upper Bound: {theoretical_max_defect_rate * 100:.2f}%")
        print("-" * 80)

    # -------------------------------------------------------------------------
    # PART A: Proposition 1 Defect Bound Evaluation: Uncalibrated vs Calibrated
    # -------------------------------------------------------------------------
    # Test across 3 latent quality regimes: Clean (20% defect), Balanced (50%), Hostile (80%)
    regimes = {
        "Clean Repo (20% Defect)": 0.20,
        "Balanced Prior (50% Defect)": 0.50,
        "Hostile/Buggy Agent (80% Defect)": 0.80,
    }
    regime_results = {}

    total_governed_cost = 0.0
    total_naive_cost = n_samples * 2.00  # $2.00 per full container matrix
    docker_containers_run = 0
    docker_containers_avoided = 0

    cost_schedule = [0.0001, 0.0005, 0.0005, 0.0020, 0.0020, 0.0020]

    for regime_name, defect_rate in regimes.items():
        n_regime = n_samples // len(regimes)

        # 1. Uncalibrated Governor (Fixed prior = 0.50, max_stages = 4)
        uncal_gov = Governor(
            reward=reward, loss=loss, prior=0.50, max_stages=4, defect_leakage=defect_leakage
        )
        uncal_admitted = 0
        uncal_defective = 0

        # 2. Calibrated Governor (Telemetry/historical prior = 1 - defect_rate, max_stages = 6)
        cal_prior = max(0.05, 1.0 - defect_rate)
        cal_gov = Governor(
            reward=reward, loss=loss, prior=cal_prior, max_stages=6, defect_leakage=defect_leakage
        )
        cal_admitted = 0
        cal_defective = 0

        for _ in range(n_regime):
            # Ground truth latent state: 1 = conforming, 0 = defective
            is_conforming = random.random() >= defect_rate

            # Simulate uncalibrated run
            b_uncal = 0.50
            pass_uncal = True
            for s in range(4):
                total_governed_cost += cost_schedule[s]
                p_pass = True if is_conforming else (random.random() < defect_leakage)
                if p_pass:
                    b_uncal = uncal_gov.posterior_belief(b_uncal, True, s)
                else:
                    pass_uncal = False
                    break
            if pass_uncal and b_uncal >= p_star:
                uncal_admitted += 1
                if not is_conforming:
                    uncal_defective += 1

            # Simulate calibrated run
            b_cal = cal_prior
            pass_cal = True
            for s in range(6):
                p_pass = True if is_conforming else (random.random() < defect_leakage)
                if p_pass:
                    b_cal = cal_gov.posterior_belief(b_cal, True, s)
                else:
                    pass_cal = False
                    break
            if pass_cal and b_cal >= p_star:
                docker_containers_run += 1
                total_governed_cost += 0.0200
                cal_admitted += 1
                if not is_conforming:
                    cal_defective += 1
            else:
                docker_containers_avoided += 1

        uncal_rate = (uncal_defective / uncal_admitted) if uncal_admitted > 0 else 0.0
        cal_rate = (cal_defective / cal_admitted) if cal_admitted > 0 else 0.0

        regime_results[regime_name] = {
            "incoming_defect_rate_pct": round(defect_rate * 100, 1),
            "candidates_evaluated": n_regime,
            "uncalibrated_prior_50": {
                "admitted": uncal_admitted,
                "defects": uncal_defective,
                "defect_rate_pct": round(uncal_rate * 100, 2),
                "bound_satisfied": uncal_rate <= theoretical_max_defect_rate,
            },
            "calibrated_governor": {
                "admitted": cal_admitted,
                "defects": cal_defective,
                "defect_rate_pct": round(cal_rate * 100, 2),
                "bound_satisfied": cal_rate <= theoretical_max_defect_rate,
                "safety_margin_pct": round((theoretical_max_defect_rate - cal_rate) * 100, 2),
            },
        }

    # -------------------------------------------------------------------------
    # PART B: Heavy-Traffic Queue Congestion Dynamics & Dynamic Review Shielding
    # -------------------------------------------------------------------------
    # Simulate arrivals lambda in {1, 2, 4, 8, 12, 16, 24, 32} against capacity K = 4
    capacity_K = 4
    review_cost_k = 1
    knapsack = KnapsackController(default_capacity=capacity_K, review_cost_k=review_cost_k)

    arrival_intensities = [1, 2, 4, 8, 12, 16, 24, 32]
    congestion_series = []

    for arrival_count in arrival_intensities:
        # High-value qualified candidate pool (beliefs between 85% and 98%)
        batch_candidates = []
        for c_idx in range(arrival_count):
            b = 0.85 + 0.13 * random.random()
            batch_candidates.append(
                CandidateSubmission(
                    candidate_id=f"PR-{arrival_count:02d}-{c_idx:02d}",
                    task_id="swe-task",
                    posterior_belief=b,
                    reward=reward,
                    loss=loss,
                )
            )

        report = knapsack.admit_batch(batch_candidates, capacity_K=capacity_K, review_cost_k=review_cost_k)

        # Test the newly implemented Section 4.1 review boundary for a marginal candidate (b = 0.70)
        # under this batch's shadow price lambda_K
        marginal_belief = 0.70
        boundary_decision = gov.evaluate_review_boundary(
            belief=marginal_belief,
            review_cost=0.002,
            shadow_price_lambda=report.shadow_price_lambda,
            review_accuracy=1.0,
        )

        congestion_series.append({
            "arrival_load": arrival_count,
            "traffic_intensity_rho": round(arrival_count / capacity_K, 2),
            "admitted_to_review": report.total_admitted,
            "deprioritized": report.total_rejected,
            "total_welfare_W": round(report.total_welfare, 4),
            "capacity_shadow_price_lambda": round(report.shadow_price_lambda, 6),
            "marginal_candidate_belief": marginal_belief,
            "marginal_candidate_decision": boundary_decision.action,
            "marginal_review_net_utility": round(boundary_decision.u_review, 6),
        })

    # -------------------------------------------------------------------------
    # PART C: Economic & Compute Reduction Metrics
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    throughput = n_samples / elapsed if elapsed > 0 else 0.0

    cost_savings = total_naive_cost - total_governed_cost
    cost_savings_pct = (cost_savings / total_naive_cost * 100.0) if total_naive_cost > 0 else 0.0
    container_avoidance_pct = (docker_containers_avoided / n_samples * 100.0) if n_samples > 0 else 0.0

    economics = {
        "naive_ci_cost_usd": round(total_naive_cost, 2),
        "governed_agent_ci_cost_usd": round(total_governed_cost, 4),
        "cost_savings_usd": round(cost_savings, 2),
        "cost_savings_pct": round(cost_savings_pct, 2),
        "docker_containers_run": docker_containers_run,
        "docker_containers_avoided": docker_containers_avoided,
        "container_avoidance_pct": round(container_avoidance_pct, 2),
    }

    if verbose:
        print("\n[1] PROPOSITION 1 SAFETY BOUND EVALUATION (Defect Bound <= 16.67%)")
        print(f"{'Regime Name':<34} {'Uncalibrated Prior 50%':<26} {'Calibrated Governor':<26} {'Bound Status'}")
        print(f"{'':<34} {'Admit (Defect %)':<26} {'Admit (Defect %)':<26} {'(Calibrated)'}")
        print("-" * 96)
        for r_name, r_data in regime_results.items():
            u = r_data["uncalibrated_prior_50"]
            c = r_data["calibrated_governor"]
            u_str = f"{u['admitted']} ({u['defect_rate_pct']:.1f}% defect)"
            c_str = f"{c['admitted']} ({c['defect_rate_pct']:.1f}% defect)"
            status_str = f"PASSED (Margin: +{c['safety_margin_pct']:.1f}%)" if c["bound_satisfied"] else "FAILED"
            print(f"{r_name:<34} {u_str:<26} {c_str:<26} [{status_str}]")

        print("\n[2] HEAVY-TRAFFIC QUEUE CONGESTION & DYNAMIC REVIEW BOUNDARY SHIELDING (K = 4 slots)")
        print(f"{'Arrivals':<10} {'Load rho':<10} {'Admitted':<10} {'Deprioritized':<15} {'Welfare W':<12} {'Shadow Price lambda':<22} {'Marginal (b=70%) Action'}")
        print("-" * 100)
        for c in congestion_series:
            print(
                f"{c['arrival_load']:<10} {c['traffic_intensity_rho']:<10.2f} {c['admitted_to_review']:<10} "
                f"{c['deprioritized']:<15} ${c['total_welfare_W']:<11.4f} ${c['capacity_shadow_price_lambda']:<21.6f} "
                f"[{c['marginal_candidate_decision']}] (U_rev=${c['marginal_review_net_utility']:.4f})"
            )

        print("\n[3] ECONOMIC COMPUTE REDUCTION (ROAD A ROI)")
        print(f"Naive Docker CI Cost:         ${economics['naive_ci_cost_usd']:,.2f} ($2.00 / candidate)")
        print(f"Governed Agent CI Cost:       ${economics['governed_agent_ci_cost_usd']:,.4f}")
        print(f"Total Cost Savings:           ${economics['cost_savings_usd']:,.2f} ({economics['cost_savings_pct']:.2f}% reduction)")
        print(f"Containers Avoided:           {economics['docker_containers_avoided']:,} / {n_samples:,} ({economics['container_avoidance_pct']:.1f}%)")
        print(f"Runtime Throughput:           {throughput:,.1f} candidates/sec ({elapsed:.3f} s total)")
        print("=" * 80)

    results = SimulationResults(
        total_candidates=n_samples,
        duration_sec=elapsed,
        throughput_per_sec=throughput,
        proposition_1_bound=regime_results,
        queue_congestion={"capacity_K": capacity_K, "series": congestion_series},
        economics=economics,
    )
    return results


def main() -> None:
    results = run_monte_carlo_simulation(n_samples=10000, verbose=True)
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "simulation_bounds_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\n[+] Empirical simulation artifact written to: {out_file}")


if __name__ == "__main__":
    main()
