"""Pillar II: Tiered Verification Pyramid.

Orchestrates a four-tier heterogeneous cost verification hierarchy:
- Tier 0: Syntactic AST gate, compile/type-check ($c_0 = $0.0000)
- Tier 1: Rapid property-based fuzzing ($c_1 ~ $0.0001)
- Tier 2: Deliberative model-authored boundary probes ($c_2 ~ $0.0050)
- Tier 3: Deep sandboxed container replay and SMT checks ($c_3 >= $0.0200)

Demonstrates that steep cost escalation breaks the policy degeneracy bound.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from governed_agent.governor import Governor


@dataclass
class ProbeOutcome:
    tier: int
    name: str
    passed: bool
    cost: float
    latency_sec: float
    diagnostic: str = ""
    execution_status: str = "OBSERVED"


@dataclass
class TieredExecutionReport:
    candidate_id: str
    passed: bool
    admitted: bool
    terminal_tier: int
    probes_executed: int
    total_cost: float
    total_latency: float
    initial_belief: float
    final_belief: float
    decision: str
    docker_invoked: bool = False
    outcomes: List[ProbeOutcome] = field(default_factory=list)

    @property
    def verification_complete(self) -> bool:
        return not any(o.execution_status in {
            'TIMEOUT', 'CRASH', 'OVERSIZED_OUTPUT', 'INFRASTRUCTURE_FAILURE', 'CANCELLED'
        } for o in self.outcomes)

    @property
    def witnessed_mismatch(self) -> bool:
        return any(o.execution_status == 'MISMATCH' for o in self.outcomes)


class TieredPyramid:
    """Orchestrates hierarchical tiered verification with economic gating."""

    DEFAULT_TIER_COSTS = {
        0: 0.0000,   # Deterministic AST / Type check
        1: 0.0001,   # Fast property fuzzing
        2: 0.0005,   # Deliberative LLM probes
        3: 0.0020,   # Isolated sandbox / SMT
    }

    def __init__(
        self,
        tier_costs: Optional[Dict[int, float]] = None,
        isolated_runner: Optional[Any] = None,
        trusted_in_process: bool = False,
        container_tier: int = 3,
    ):
        self.tier_costs = dict(self.DEFAULT_TIER_COSTS)
        if tier_costs:
            self.tier_costs.update(tier_costs)
        self.trusted_in_process = trusted_in_process
        self.container_tier = container_tier
        if trusted_in_process and isolated_runner is not None:
            raise ValueError('Select trusted fixtures or isolated execution, not both')
        if isolated_runner is None and not trusted_in_process:
            from governed_agent.isolation import ProcessIsolatedRunner
            isolated_runner = ProcessIsolatedRunner()
        self.isolated_runner = isolated_runner

    def execute_syntactic_gate(self, code_str: str) -> Tuple[bool, str]:
        """Tier 0: Pure AST syntax validation."""
        try:
            tree = ast.parse(code_str)
            # Check basic structural invariants
            has_func = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            if not has_func:
                return False, "Tier 0 Failure: No function definition found in AST"
            return True, "Tier 0 Passed: Valid AST structure"
        except SyntaxError as ex:
            return False, f"Tier 0 SyntaxError: {ex}"
        except Exception as ex:
            return False, f"Tier 0 Error: {ex}"

    def run_tiered_pipeline(
        self,
        candidate_id: str,
        candidate_code: str,
        candidate_fn: Callable[..., Any],
        reference_fn: Callable[..., Any],
        tier1_probes: List[Dict[str, Any]],
        tier2_probes: List[Dict[str, Any]],
        tier3_probes: Optional[List[Dict[str, Any]]] = None,
        governor: Optional[Governor] = None,
        isolated_runner: Optional[Any] = None,
    ) -> TieredExecutionReport:
        """Executes verification through ascending cost tiers with dynamic gating."""
        effective_isolated_runner = isolated_runner if isolated_runner is not None else self.isolated_runner
        if governor is None:
            # Heterogeneous cost schedule matching the tiers
            costs = [self.tier_costs[1], self.tier_costs[2], self.tier_costs[2], self.tier_costs[3]]
            governor = Governor(
                reward=0.02,
                loss=0.10,
                prior=0.50,
                max_stages=len(costs),
                cost_schedule=costs,
                defect_leakage=0.6125,
            )

        outcomes: List[ProbeOutcome] = []
        total_cost = 0.0
        total_latency = 0.0
        current_belief = governor.prior
        consecutive_passes = 0
        stage_idx = 0

        # -------------------------------------------------------------
        # Tier 0: Syntactic Gate ($c_0 = $0.0000)
        # -------------------------------------------------------------
        t0_start = time.perf_counter()
        t0_passed, t0_diag = self.execute_syntactic_gate(candidate_code)
        t0_latency = time.perf_counter() - t0_start
        c0 = self.tier_costs[0]
        total_cost += c0
        total_latency += t0_latency
        outcomes.append(ProbeOutcome(tier=0, name="ast_syntax_gate", passed=t0_passed, cost=c0, latency_sec=t0_latency, diagnostic=t0_diag))

        if not t0_passed:
            return TieredExecutionReport(
                candidate_id=candidate_id,
                passed=False,
                admitted=False,
                terminal_tier=0,
                probes_executed=1,
                total_cost=total_cost,
                total_latency=total_latency,
                initial_belief=governor.prior,
                final_belief=0.0,
                decision="HALT_AND_REJECT",
                outcomes=outcomes,
            )

        def stop_if_governor_halts() -> Optional[TieredExecutionReport]:
            """Honor the policy and finite horizon before every paid probe."""
            decision = governor.evaluate_state(stage_idx, consecutive_passes)
            if decision.action == "CONTINUE" and stage_idx < governor.max_stages:
                return None
            admitted = (
                decision.action == "HALT_AND_COMMIT"
                and current_belief >= governor.p_star
            )
            return TieredExecutionReport(
                candidate_id=candidate_id,
                passed=admitted,
                admitted=admitted,
                terminal_tier=outcomes[-1].tier,
                probes_executed=len(outcomes),
                total_cost=total_cost,
                total_latency=total_latency,
                initial_belief=governor.prior,
                final_belief=current_belief,
                decision="HALT_AND_COMMIT" if admitted else "HALT_AND_REJECT",
                docker_invoked=any(o.tier >= self.container_tier and effective_isolated_runner is not None for o in outcomes),
                outcomes=outcomes,
            )

        # -------------------------------------------------------------
        # Helper runner for dynamic probe execution
        # -------------------------------------------------------------
        def run_probe_call(tier: int, probe_name: str, kwargs: Dict[str, Any]) -> bool:
            nonlocal total_cost, total_latency, current_belief, consecutive_passes, stage_idx
            cost = self.tier_costs[tier]
            p_start = time.perf_counter()
            from copy import deepcopy

            def observe(fn):
                try:
                    return None, fn(**deepcopy(kwargs))
                except Exception as exc:
                    return type(exc), None

            execution_status = 'OBSERVED'
            use_container = bool(effective_isolated_runner is not None and tier >= self.container_tier and candidate_code)
            if use_container:
                # Isolated out-of-process candidate observation (Docker Sandbox)
                func_name = getattr(candidate_fn, "__name__", None)
                if func_name == "<lambda>":
                    func_name = None
                from governed_agent.isolation import IsolatedExecutionResult, ExecutionStatus
                try:
                    cand_exec = effective_isolated_runner.run_candidate(
                        code=candidate_code, func_name=func_name, kwargs=deepcopy(kwargs))
                except Exception:
                    cand_exec = IsolatedExecutionResult(ExecutionStatus.INFRASTRUCTURE_FAILURE,
                                                       diagnostic='Isolation backend failed')
                # Observe reference in trusted process
                if cand_exec.status in ('SUCCESS', 'EXCEPTION'):
                    ref_err, ref_res = observe(reference_fn)
                else:
                    ref_err, ref_res = None, None
                ref_err_name = ref_err.__module__+'.'+ref_err.__qualname__ if ref_err is not None else None

                if cand_exec.status == "SUCCESS":
                    if ref_err_name is not None:
                        p_pass = False
                        p_diag = f"One-sided exception: candidate returned normally, reference raised {ref_err_name}"
                    else:
                        p_pass = (cand_exec.value == ref_res)
                        p_diag = "Pass" if p_pass else f"Mismatch: cand={cand_exec.value!r} vs ref={ref_res!r}"
                elif cand_exec.status == "EXCEPTION":
                    cand_err_name = cand_exec.exception_type
                    if ref_err_name is not None and cand_err_name == ref_err_name:
                        p_pass = True
                        p_diag = "Matching Exception"
                    else:
                        p_pass = False
                        p_diag = f"Exception Mismatch: cand={cand_err_name} vs ref={ref_err_name}"
                else:
                    # TIMEOUT, CRASH, OVERSIZED_OUTPUT, INFRASTRUCTURE_FAILURE
                    p_pass = False
                    p_diag = f"Isolation failure [{cand_exec.status}]: {cand_exec.diagnostic}"
                    execution_status = str(getattr(cand_exec.status, 'value', cand_exec.status))
                    if execution_status not in {s.value for s in ExecutionStatus}:
                        execution_status = 'INFRASTRUCTURE_FAILURE'
            elif self.trusted_in_process or (effective_isolated_runner is not None and tier < self.container_tier):
                # Road A Host pre-check evaluation (in-process)
                active_cand_fn = candidate_fn
                if active_cand_fn is None or getattr(active_cand_fn, "__name__", "") == "dummy_isolated_callable":
                    if candidate_code:
                        try:
                            ns: Dict[str, Any] = {}
                            exec(candidate_code, ns)
                            f_name = getattr(candidate_fn, "__name__", None)
                            if not f_name or f_name == "dummy_isolated_callable":
                                tree = ast.parse(candidate_code)
                                funcs = [n.name for n in getattr(tree, "body", []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                                f_name = funcs[0] if funcs else None
                            if f_name and f_name in ns:
                                active_cand_fn = ns[f_name]
                        except Exception:
                            pass
                cand_err, cand_res = observe(active_cand_fn)
                ref_err, ref_res = observe(reference_fn)
                if cand_err is not None or ref_err is not None:
                    p_pass = cand_err is not None and cand_err is ref_err
                    p_diag = "Matching Exception" if p_pass else "Exception Mismatch"
                else:
                    p_pass = (cand_res == ref_res)
                    p_diag = "Pass" if p_pass else f"Mismatch: cand={cand_res!r} vs ref={ref_res!r}"
            else:
                p_pass = False
                execution_status = 'INFRASTRUCTURE_FAILURE'
                p_diag = 'Isolated execution required; no in-process fallback'

            if execution_status == 'OBSERVED':
                execution_status = 'MATCH' if p_pass else 'MISMATCH'

            p_lat = time.perf_counter() - p_start
            total_cost += cost
            total_latency += p_lat
            outcomes.append(ProbeOutcome(tier=tier, name=probe_name, passed=p_pass, cost=cost,
                latency_sec=p_lat, diagnostic=p_diag, execution_status=execution_status))

            if p_pass:
                consecutive_passes += 1
                current_belief = governor.posterior_belief(current_belief, True, stage_idx)
            elif execution_status == 'MISMATCH':
                current_belief = 0.0

            stage_idx += 1
            return p_pass

        # -------------------------------------------------------------
        # Tier 1: Property Fuzzing ($c_1 ~ $0.0001)
        # -------------------------------------------------------------
        for idx, probe in enumerate(tier1_probes):
            halted_report = stop_if_governor_halts()
            if halted_report is not None:
                return halted_report
            p_name = probe.get("name", f"tier1_fuzz_{idx}")
            kwargs = probe.get("input", {})
            passed = run_probe_call(1, p_name, kwargs)
            if not passed:
                return TieredExecutionReport(
                    candidate_id=candidate_id,
                    passed=False,
                    admitted=False,
                    terminal_tier=1,
                    probes_executed=len(outcomes),
                    total_cost=total_cost,
                    total_latency=total_latency,
                    initial_belief=governor.prior,
                    final_belief=current_belief,
                    decision="HALT_AND_REJECT",
                    outcomes=outcomes,
                )

        # -------------------------------------------------------------
        # Tier 2: Deliberative Probing ($c_2 ~ $0.0050)
        # -------------------------------------------------------------
        for idx, probe in enumerate(tier2_probes):
            halted_report = stop_if_governor_halts()
            if halted_report is not None:
                return halted_report

            p_name = probe.get("name", f"tier2_probe_{idx}")
            kwargs = probe.get("input", {})
            passed = run_probe_call(2, p_name, kwargs)
            if not passed:
                return TieredExecutionReport(
                    candidate_id=candidate_id,
                    passed=False,
                    admitted=False,
                    terminal_tier=2,
                    probes_executed=len(outcomes),
                    total_cost=total_cost,
                    total_latency=total_latency,
                    initial_belief=governor.prior,
                    final_belief=current_belief,
                    decision="HALT_AND_REJECT",
                    outcomes=outcomes,
                )

        # -------------------------------------------------------------
        # Tier 3: Isolated Sandboxing / SMT ($c_3 >= $0.0200)
        # -------------------------------------------------------------
        if tier3_probes:
            for idx, probe in enumerate(tier3_probes):
                halted_report = stop_if_governor_halts()
                if halted_report is not None:
                    return halted_report
                p_name = probe.get("name", f"tier3_sandbox_{idx}")
                kwargs = probe.get("input", {})
                passed = run_probe_call(3, p_name, kwargs)
                if not passed:
                    return TieredExecutionReport(
                        candidate_id=candidate_id,
                        passed=False,
                        admitted=False,
                        terminal_tier=3,
                        probes_executed=len(outcomes),
                        total_cost=total_cost,
                        total_latency=total_latency,
                        initial_belief=governor.prior,
                        final_belief=current_belief,
                        decision="HALT_AND_REJECT",
                        docker_invoked=True,
                        outcomes=outcomes,
                    )

        # Final decision
        final_decision = governor.evaluate_state(stage_idx, consecutive_passes)
        is_admitted = (current_belief >= governor.p_star) and (final_decision.action == "HALT_AND_COMMIT")

        return TieredExecutionReport(
            candidate_id=candidate_id,
            passed=is_admitted,
            admitted=is_admitted,
            terminal_tier=outcomes[-1].tier if outcomes else 0,
            probes_executed=len(outcomes),
            total_cost=total_cost,
            total_latency=total_latency,
            initial_belief=governor.prior,
            final_belief=current_belief,
            decision="HALT_AND_COMMIT" if is_admitted else "HALT_AND_REJECT",
            docker_invoked=any(o.tier >= self.container_tier and effective_isolated_runner is not None for o in outcomes),
            outcomes=outcomes,
        )
