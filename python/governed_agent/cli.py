"""Command-line interface for the Governed Agent runtime."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional

from governed_agent.governor import Governor
from governed_agent.knapsack import CandidateSubmission, KnapsackController
from governed_agent.mutator import MutationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="governed-agent",
        description="The Governed Agent: Economically Regulated Decision-Theoretic Runtime"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Verify subcommand
    verify_p = subparsers.add_parser("verify", help="Run 5-pillar verification pipeline on candidate Python code")
    verify_p.add_argument("candidate", type=str, help="Path to candidate Python file (.py)")
    verify_p.add_argument("--function", type=str, default=None, help="Target function name inside candidate file (auto-detected if omitted)")
    verify_p.add_argument("--reference", type=str, default=None, help="Path to reference Python file (.py)")
    verify_p.add_argument("--reference-function", type=str, default=None, help="Reference function name (defaults to target function)")
    verify_p.add_argument("--probes", type=str, default=None, help="Path to JSON file containing probe suite")
    verify_p.add_argument("--isolated", dest="isolated", action="store_true", default=True, help="Execute candidate inside disposable Docker container sandbox (default: True)")
    verify_p.add_argument("--trusted", dest="isolated", action="store_false", help="Execute candidate in-process without container isolation (trusted candidates only)")
    verify_p.add_argument("--reward", type=float, default=0.02, help="Reward value R ($)")
    verify_p.add_argument("--loss", type=float, default=0.10, help="Loss penalty L ($)")
    verify_p.add_argument("--prior", type=float, default=0.50, help="Initial prior belief p0")
    verify_p.add_argument("--stages", type=int, default=4, help="Max verification stages")
    verify_p.add_argument("--output", type=str, default=None, help="Optional path to output verification receipt JSON")

    # Audit subcommand
    audit_p = subparsers.add_parser("audit", help="Audit AST mutation score on a Python file")
    audit_p.add_argument("file", type=str, help="Path to Python file to mutate and audit")
    audit_p.add_argument("--max-mutants", type=int, default=10, help="Maximum mutants to generate")

    # Triage subcommand
    triage_p = subparsers.add_parser("triage", help="Evaluate Bayesian Bellman stopping policy")
    triage_p.add_argument("--reward", type=float, default=0.02, help="Reward value R ($)")
    triage_p.add_argument("--loss", type=float, default=0.10, help="Loss penalty L ($)")
    triage_p.add_argument("--prior", type=float, default=0.50, help="Initial prior belief p0")
    triage_p.add_argument("--stages", type=int, default=4, help="Max verification stages")

    # Queue subcommand
    queue_p = subparsers.add_parser("queue", help="Run Pillar V Knapsack queue admission on candidates JSON or institutional dataset")
    queue_p.add_argument("file", type=str, nargs="?", default=None, help="JSON file containing candidate list")
    queue_p.add_argument("--dataset", type=str, choices=["openreview", "nih", "uspto"], default=None, help="Named institutional queue dataset")
    queue_p.add_argument("--capacity", type=float, default=3.0, help="Review queue capacity K (slots or hours)")

    # Demo subcommand
    demo_p = subparsers.add_parser("demo", help="Run full 5-pillar end-to-end demonstration harness")
    demo_p.add_argument("--isolated", action="store_true", default=False, help="Run candidate evaluation under real container isolation")

    # Gate subcommand (Road A: Tiered Verification Gate)
    gate_p = subparsers.add_parser("gate", help="Run Road A Tiered Verification Gate (Host Pre-Checks -> Docker Container)")
    gate_p.add_argument("candidate", type=str, help="Path to candidate Python file (.py) or pull requests JSON file")
    gate_p.add_argument("--reference", type=str, default=None, help="Path to reference Python file (.py)")
    gate_p.add_argument("--probes", type=str, default=None, help="Path to JSON file containing probe suite")
    gate_p.add_argument("--capacity", type=int, default=3, help="Review queue capacity K for batch gating")
    gate_p.add_argument("--docker", dest="docker", action="store_true", default=False, help="Enable live Docker container sandbox for Tier 3 (default: dry-run simulated sandbox)")
    gate_p.add_argument("--output", type=str, default=None, help="Optional path to output gate receipt or batch report JSON")

    # Info subcommand
    subparsers.add_parser("info", help="Display runtime version and Rust acceleration status")

    return parser


def handle_info(args: argparse.Namespace) -> int:
    try:
        from . import HAS_RUST_CORE, __version__
    except Exception:
        HAS_RUST_CORE = False
        __version__ = "0.2.0"
    backend = "Rust Native Core (governed_agent_core)" if HAS_RUST_CORE else "Python Fallback"
    print("=" * 60)
    print("  THE GOVERNED AGENT: RUNTIME INFORMATION")
    print("=" * 60)
    print(f"Package Version:       {__version__}")
    print(f"Acceleration Engine:   {backend}")
    print(f"Rust Core Active:      {HAS_RUST_CORE}")
    print(f"Road A Gatekeeper:     Active (Host Pre-Checks -> Docker Sandbox)")
    print("=" * 60)
    return 0


def handle_audit(args: argparse.Namespace) -> int:
    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1
    with open(args.file, "r", encoding="utf-8") as f:
        code = f.read()

    engine = MutationEngine(qualification_threshold=0.80)
    mutants = engine.generate_mutants(code, max_mutants=args.max_mutants)
    print(f"Auditing '{args.file}': Generated {len(mutants)} AST mutants")
    print("-" * 60)
    for idx, m in enumerate(mutants, 1):
        print(f"[{idx}] {m.operator} (Line {m.lineno}): {m.description}")
    print("-" * 60)
    print("Run test suite against generated mutants to calculate Mutation Score MS.")
    return 0


def handle_triage(args: argparse.Namespace) -> int:
    gov = Governor(
        reward=args.reward,
        loss=args.loss,
        prior=args.prior,
        max_stages=args.stages
    )
    decision = gov.evaluate_state(0, 0)
    p_star = gov.p_star
    print("=" * 60)
    print("  BAYESIAN BELLMAN GOVERNOR CONFIGURATION")
    print("=" * 60)
    print(f"Reward R: ${args.reward:.4f} | Loss L: ${args.loss:.4f} | Loss Stakes: {args.loss / args.reward:.1f}x")
    print(f"Prior Belief: {args.prior*100:.1f}% | Critical Threshold p*: {p_star*100:.2f}%")
    print(f"Backward Induction Continuation Value: ${decision.expected_utility:.6f}")
    print(f"Optimal Action at Stage 0: {decision.action} (VOI = ${decision.voi:.6f})")
    print("=" * 60)
    return 0


def handle_queue(args: argparse.Namespace) -> int:
    if args.dataset:
        from evaluate_institutional_queues import (
            load_openreview_data,
            load_nih_data,
            load_uspto_data,
        )
        if args.dataset == "openreview":
            raw_cands = load_openreview_data()
        elif args.dataset == "nih":
            raw_cands = load_nih_data()
        elif args.dataset == "uspto":
            raw_cands = load_uspto_data()
        else:
            print(f"Error: unknown dataset: {args.dataset}", file=sys.stderr)
            return 1

        candidates = [
            CandidateSubmission(
                candidate_id=c["candidate_id"],
                task_id=args.dataset,
                posterior_belief=c["quality_p"],
                reward=0.02,
                loss=0.10,
                raw_output=c,
            )
            for c in raw_cands
        ]
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1
        with open(args.file, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        candidates = [
            CandidateSubmission(
                candidate_id=item.get("candidate_id", f"PR-{i}"),
                task_id=item.get("task_id", f"task-{i}"),
                posterior_belief=item.get("posterior_belief", 0.90),
                reward=item.get("reward", 0.02),
                loss=item.get("loss", 0.10),
                review_cost=item.get("review_cost", item.get("review_cost_k", 1.0)),
            )
            for i, item in enumerate(raw_items)
        ]
    else:
        print("Error: must specify a JSON candidate file or --dataset {openreview,nih,uspto}", file=sys.stderr)
        return 1

    controller = KnapsackController(default_capacity=args.capacity)
    report = controller.admit_batch(candidates)
    source_label = f"Dataset: {args.dataset}" if args.dataset else f"File: {args.file}"
    has_het_costs = any(c.review_cost is not None and abs(c.review_cost - 1.0) > 1e-6 for c in candidates)
    print("=" * 60)
    print(f"  PILLAR V KNAPSACK QUEUE ADMISSION ({source_label})")
    print("=" * 60)
    print(f"Review Capacity K:  {args.capacity} slots/hours")
    print(f"Total Candidates:   {report.total_candidates}")
    print(f"Optimal Depth m*:   {report.optimal_selection_depth}")
    print(f"Total Admitted:     {report.total_admitted}")
    print(f"Total Rejected:     {report.total_rejected}")
    if has_het_costs or report.total_admitted_cost > 0.0:
        print(f"Capacity Consumed:  {report.total_admitted_cost:.2f} / {report.capacity_K:.2f} (Remaining: {report.remaining_capacity:.2f})")
    print(f"Total Welfare W:    ${report.total_welfare:.4f}")
    print(f"Capacity Shadow Price lambda_K: ${report.shadow_price_lambda:.6f}")
    print("=" * 60)
    if report.admitted:
        print("Admitted Candidates:")
        for c in report.admitted:
            title = c.raw_output.get("title", "") if isinstance(c.raw_output, dict) else ""
            title_str = f" - '{title[:35]}...'" if title else ""
            cost_val = c.review_cost if c.review_cost is not None else 1.0
            cost_str = f", cost = {cost_val:.2f}, rho = {c.density:+.4f}" if has_het_costs else ""
            print(f"  [+] {c.candidate_id:25s} (p = {c.posterior_belief*100:.1f}%, E[U] = ${c.expected_utility:+.4f}{cost_str}){title_str}")
    if report.rejected:
        print("Rejected Candidates:")
        for c in report.rejected:
            title = c.raw_output.get("title", "") if isinstance(c.raw_output, dict) else ""
            title_str = f" - '{title[:35]}...'" if title else ""
            cost_val = c.review_cost if c.review_cost is not None else 1.0
            cost_str = f", cost = {cost_val:.2f}, rho = {c.density:+.4f}" if has_het_costs else ""
            print(f"  [-] {c.candidate_id:25s} (p = {c.posterior_belief*100:.1f}%, E[U] = ${c.expected_utility:+.4f}{cost_str}){title_str}")
    print("=" * 60)
    return 0


def handle_verify(args: argparse.Namespace) -> int:
    from governed_agent.pyramid import TieredPyramid

    if not os.path.exists(args.candidate):
        print(f"Error: Candidate file not found: {args.candidate}", file=sys.stderr)
        return 1

    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate_code = f.read()

    # Step 1: Detect target function name and validate syntax
    try:
        cand_tree = ast.parse(candidate_code)
    except SyntaxError as ex:
        # Tier 0 instant syntactic failure
        print("=" * 80)
        print("  THE GOVERNED AGENT: CANDIDATE VERIFICATION RECEIPT")
        print("=" * 80)
        print(f"Candidate File:     {args.candidate}")
        print(f"Syntax Check:       FAILED -> {ex}")
        print(f"Decision:           HALT_AND_REJECT (Tier 0 Syntax Gate)")
        print(f"Verification Gate:  REJECTED | Final Belief: 0.0%")
        print("=" * 80)
        if args.output:
            receipt = {
                "candidate_file": args.candidate,
                "passed": False,
                "admitted": False,
                "terminal_tier": 0,
                "decision": "HALT_AND_REJECT",
                "diagnostic": f"SyntaxError: {ex}",
            }
            with open(args.output, "w", encoding="utf-8") as out_f:
                json.dump(receipt, out_f, indent=2)
        return 1

    func_name = args.function
    if not func_name:
        cand_funcs = [
            n.name for n in getattr(cand_tree, "body", [])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not cand_funcs:
            for n in ast.walk(cand_tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cand_funcs.append(n.name)
        if not cand_funcs:
            print(f"Error: No function definitions found in {args.candidate}", file=sys.stderr)
            return 1
        public_funcs = [f for f in cand_funcs if not f.startswith("_")]
        func_name = public_funcs[0] if public_funcs else cand_funcs[0]

    # Step 2: Load Reference Function
    ref_fn = None
    ref_name = args.reference_function or func_name
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
            return 1
        with open(args.reference, "r", encoding="utf-8") as f:
            ref_code = f.read()
        ref_ns: Dict[str, Any] = {}
        exec(ref_code, ref_ns)
        ref_fn = ref_ns.get(ref_name)
        if ref_fn is None:
            print(f"Error: Reference function '{ref_name}' not found in {args.reference}", file=sys.stderr)
            return 1

    # Step 3: Load or Synthesize Probes
    loaded_probes: List[Dict[str, Any]] = []
    if args.probes:
        if not os.path.exists(args.probes):
            print(f"Error: Probes file not found: {args.probes}", file=sys.stderr)
            return 1
        with open(args.probes, "r", encoding="utf-8") as f:
            probes_data = json.load(f)
        if isinstance(probes_data, list):
            loaded_probes = probes_data
        elif isinstance(probes_data, dict):
            if "probes" in probes_data and isinstance(probes_data["probes"], list):
                loaded_probes = probes_data["probes"]
            elif "families" in probes_data and isinstance(probes_data["families"], list):
                for fam in probes_data["families"]:
                    loaded_probes.extend(fam.get("probes", []))
            else:
                print("Error: Unrecognized probes JSON structure.", file=sys.stderr)
                return 1

    if not loaded_probes and ref_fn is not None:
        loaded_probes = [
            {"name": "default_empty", "input": {}, "tier": 1},
        ]

    if not loaded_probes and ref_fn is None:
        print("Error: Either --reference or --probes must be provided.", file=sys.stderr)
        return 1

    # If reference function is not provided from file, check if probes have expected outputs
    if ref_fn is None:
        has_expectations = all(("expected" in p or "output" in p) for p in loaded_probes)
        if not has_expectations:
            print("Error: Without a --reference file, every probe in --probes must define an 'expected' value.", file=sys.stderr)
            return 1
        class ProbeExpectationOracle:
            def __init__(self, plist: List[Dict[str, Any]]):
                self.table: Dict[str, Any] = {}
                for p in plist:
                    inp = p.get("input") or p.get("proposed_test_input") or {}
                    key = json.dumps(inp, sort_keys=True)
                    self.table[key] = p.get("expected", p.get("output"))
            def __call__(self, **kwargs: Any) -> Any:
                key = json.dumps(kwargs, sort_keys=True)
                if key in self.table:
                    return self.table[key]
                raise KeyError(f"No expectation for {kwargs}")
        ref_fn = ProbeExpectationOracle(loaded_probes)
        ref_fn.__name__ = func_name

    # Separate probes into tier 1 and tier 2
    tier1_probes: List[Dict[str, Any]] = []
    tier2_probes: List[Dict[str, Any]] = []
    for idx, p in enumerate(loaded_probes):
        p_name = p.get("name") or p.get("obligation") or f"probe_{idx}"
        p_input = p.get("input") or p.get("proposed_test_input") or {}
        p_tier = p.get("tier", 1 if idx == 0 else 2)
        probe_dict = {"name": p_name, "input": p_input}
        if p_tier == 1:
            tier1_probes.append(probe_dict)
        else:
            tier2_probes.append(probe_dict)

    if not tier1_probes and tier2_probes:
        tier1_probes.append(tier2_probes.pop(0))

    # Candidate callable for in-process or dummy callable for isolation
    cand_fn = None
    if not args.isolated:
        cand_ns: Dict[str, Any] = {}
        exec(candidate_code, cand_ns)
        cand_fn = cand_ns.get(func_name)
        if cand_fn is None:
            print(f"Error: Candidate function '{func_name}' not found in {args.candidate}", file=sys.stderr)
            return 1
    else:
        def dummy_isolated_callable(**kwargs: Any) -> Any:
            pass
        dummy_isolated_callable.__name__ = func_name
        cand_fn = dummy_isolated_callable

    gov = Governor(
        reward=args.reward,
        loss=args.loss,
        prior=args.prior,
        max_stages=args.stages,
        cost_schedule=[0.0001, 0.0005, 0.0005, 0.0020][:args.stages],
        defect_leakage=0.5875,
    )
    try:
        pyramid = TieredPyramid(trusted_in_process=(not args.isolated))
    except Exception as ex:
        print(f"Error initializing verification pyramid: {ex}", file=sys.stderr)
        return 1

    report = pyramid.run_tiered_pipeline(
        candidate_id=os.path.basename(args.candidate),
        candidate_code=candidate_code,
        candidate_fn=cand_fn,
        reference_fn=ref_fn,
        tier1_probes=tier1_probes,
        tier2_probes=tier2_probes,
        governor=gov,
    )

    mode_str = "DOCKER CONTAINER (Linux Sandbox)" if args.isolated else "TRUSTED IN-PROCESS"
    status_str = "QUALIFIED & ADMITTED" if report.admitted else "REJECTED"
    print("=" * 80)
    print("  THE GOVERNED AGENT: CANDIDATE VERIFICATION RECEIPT")
    print("=" * 80)
    print(f"Candidate File:     {args.candidate}")
    print(f"Target Function:    {func_name}")
    print(f"Execution Mode:     {mode_str}")
    print(f"Economic Stakes:    Reward = ${args.reward:.4f} | Loss = ${args.loss:.4f} | Ratio = {args.loss/args.reward:.1f}x")
    print(f"Posterior Belief:   {report.initial_belief*100:.1f}% -> {report.final_belief*100:.2f}% (Critical p* = {gov.p_star*100:.2f}%)")
    print(f"Verification Gate:  {status_str}")
    print(f"Decision:           {report.decision} (Terminal Tier: {report.terminal_tier})")
    print(f"Probes Executed:    {report.probes_executed} probes | Total Cost: ${report.total_cost:.4f} | Latency: {report.total_latency:.4f}s")
    print("-" * 80)
    print("Probe Execution Outcomes:")
    for outcome in report.outcomes:
        pass_marker = "PASSED" if outcome.passed else "FAILED"
        print(f"  [Tier {outcome.tier}] {outcome.name:<30}: {pass_marker} (${outcome.cost:.4f}, {outcome.latency_sec:.4f}s) -> {outcome.diagnostic}")
    print("=" * 80)

    if args.output:
        receipt_data = {
            "candidate_file": args.candidate,
            "function_name": func_name,
            "isolated": args.isolated,
            "reward": args.reward,
            "loss": args.loss,
            "prior": args.prior,
            "passed": report.passed,
            "admitted": report.admitted,
            "terminal_tier": report.terminal_tier,
            "decision": report.decision,
            "initial_belief": report.initial_belief,
            "final_belief": report.final_belief,
            "p_star": gov.p_star,
            "probes_executed": report.probes_executed,
            "total_cost": report.total_cost,
            "total_latency": report.total_latency,
            "outcomes": [
                {
                    "tier": o.tier,
                    "name": o.name,
                    "passed": o.passed,
                    "cost": o.cost,
                    "latency_sec": o.latency_sec,
                    "diagnostic": o.diagnostic,
                    "execution_status": o.execution_status,
                }
                for o in report.outcomes
            ],
        }
        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(receipt_data, out_f, indent=2)
        print(f"Receipt written to: {args.output}")

    return 0 if report.admitted else 1


def handle_gate(args: argparse.Namespace) -> int:
    from governed_agent.gate import TieredVerificationGate
    from governed_agent.isolation import ProcessIsolatedRunner

    if not os.path.exists(args.candidate):
        print(f"Error: Candidate file not found: {args.candidate}", file=sys.stderr)
        return 1

    isolated_runner = None
    if args.docker:
        try:
            isolated_runner = ProcessIsolatedRunner()
        except Exception as ex:
            print(f"Warning: Docker runner initialization failed: {ex}. Using dry-run mode.", file=sys.stderr)

    gate = TieredVerificationGate(
        isolated_runner=isolated_runner,
        dry_run=(isolated_runner is None),
    )

    # Check if candidate is JSON (batch of pull requests / tasks)
    if args.candidate.endswith(".json"):
        with open(args.candidate, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
        if not isinstance(candidates_data, list):
            print("Error: Batch JSON file must contain a list of candidate dictionaries.", file=sys.stderr)
            return 1

        batch_report = gate.gate_batch(
            candidates_data,
            capacity_K=args.capacity,
            batch_id=os.path.basename(args.candidate),
        )

        funnel = batch_report.funnel
        econ = batch_report.economics

        print("=" * 80)
        print("  ROAD A: TIERED VERIFICATION GATE BATCH REPORT")
        print("=" * 80)
        print(f"Batch Source:           {args.candidate}")
        print(f"Total PRs Received:     {batch_report.total_candidates}")
        print(f"Queue Capacity K:       {batch_report.queue_capacity_K} review slots")
        print("-" * 80)
        print("  SCREENING FUNNEL (HOST PRE-CHECKS & DOCKER ESCALATION)")
        print(f"  [Tier 0 AST Syntax Screened]       {funnel['tier_0_syntax_screened']} PRs (instant zero-cost rejection)")
        print(f"  [Tier 1 Fast Invariants Screened]   {funnel['tier_1_crude_bug_screened']} PRs (crude bugs halted on host)")
        print(f"  [Tier 2 Mutation Oracles Screened] {funnel['tier_2_mutation_screened']} PRs (subtle bugs halted on host)")
        print(f"  [Tier 3 Qualified for Docker]      {funnel['tier_3_qualified']} PRs")
        print(f"  [Knapsack Admitted to Review]      {funnel['knapsack_admitted']} PRs")
        print(f"  [Knapsack Deprioritized]           {funnel['knapsack_deprioritized']} PRs")
        print("-" * 80)
        print("  ECONOMIC & COMPUTE REDUCTION (ROAD A ROI)")
        print(f"  Naive Full CI Cost (Docker all):   ${econ['naive_full_ci_cost_usd']:.2f}")
        print(f"  Governed Agent CI Cost:            ${econ['total_governed_cost_usd']:.4f}")
        print(f"  Total Cost Savings:                ${econ['cost_savings_usd']:.2f} ({econ['cost_savings_pct']:.1f}% savings)")
        print(f"  Docker Container Runs Avoided:     {econ['docker_invocations_avoided']}/{batch_report.total_candidates} ({econ['docker_invocations_avoided_pct']:.1f}%)")
        print(f"  Review Capacity Shadow Price lambda_K:  ${econ['capacity_shadow_price_lambda']:.6f}")
        print("=" * 80)

        if batch_report.admitted_candidates:
            print("Admitted Pull Requests (Sent to Human Review):")
            for c in batch_report.admitted_candidates:
                cid = c.get("candidate_id", c.get("pr_id", "PR"))
                title = c.get("title", "")
                title_str = f" - '{title[:40]}...'" if title else ""
                print(f"  [+] {cid:20s}{title_str}")
            print("=" * 80)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as out_f:
                json.dump(batch_report.to_dict(), out_f, indent=2)
            print(f"Batch report written to: {args.output}")

        return 0

    # Otherwise candidate is Python code file (.py)
    with open(args.candidate, "r", encoding="utf-8") as f:
        code_str = f.read()

    ref_fn = None
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
            return 1
        with open(args.reference, "r", encoding="utf-8") as f:
            ref_code = f.read()
        ref_ns: Dict[str, Any] = {}
        exec(ref_code, ref_ns)
        tree = ast.parse(ref_code)
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if funcs:
            ref_fn = ref_ns.get(funcs[0])

    loaded_probes = None
    if args.probes:
        if not os.path.exists(args.probes):
            print(f"Error: Probes file not found: {args.probes}", file=sys.stderr)
            return 1
        with open(args.probes, "r", encoding="utf-8") as f:
            loaded_probes = json.load(f)

    receipt = gate.verify_candidate(
        candidate_id=os.path.basename(args.candidate),
        candidate_code=code_str,
        reference_fn=ref_fn,
        probes=loaded_probes,
    )

    docker_status = "INVOKED" if receipt.docker_invoked else "AVOIDED (Host Short-Circuit)"
    status_str = "QUALIFIED & ADMITTED" if receipt.admitted else "HALT_AND_REJECT"
    print("=" * 80)
    print("  ROAD A: TIERED VERIFICATION GATE RECEIPT")
    print("=" * 80)
    print(f"Candidate File:     {args.candidate}")
    print(f"Verification Gate:  {status_str} (Terminal Tier: {receipt.terminal_tier})")
    print(f"Short-Circuited:    {receipt.short_circuited} (Docker Container {docker_status})")
    print(f"Belief Trajectory:  {receipt.initial_belief*100:.1f}% -> {receipt.final_belief*100:.2f}% (Critical p* = {receipt.p_star*100:.2f}%)")
    print(f"Governed Cost:      ${receipt.total_cost:.4f} (vs Naive CI ${receipt.naive_cost:.2f} -> {receipt.cost_savings_pct:.1f}% Savings)")
    print(f"Total Latency:      {receipt.total_latency_sec*1000:.2f} ms | Probes Executed: {receipt.probes_executed}")
    print("-" * 80)
    print("Tier Execution Outcomes:")
    for o in receipt.outcomes:
        res = "PASSED" if o.passed else "FAILED"
        print(f"  [Tier {o.tier}] {o.name:<30}: {res} (${o.cost:.4f}, {o.latency_sec*1000:.2f}ms) -> {o.diagnostic}")
    print("=" * 80)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(receipt.to_dict(), out_f, indent=2)
        print(f"Receipt written to: {args.output}")

    return 0 if receipt.admitted else 1


def handle_demo(args: argparse.Namespace) -> int:
    from run_governed_agent_demo import run_demo
    isolated = getattr(args, "isolated", False)
    run_demo(verbose=True, isolated=isolated)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "info":
        return handle_info(args)
    elif args.command == "gate":
        return handle_gate(args)
    elif args.command == "verify":
        return handle_verify(args)
    elif args.command == "audit":
        return handle_audit(args)
    elif args.command == "triage":
        return handle_triage(args)
    elif args.command == "queue":
        return handle_queue(args)
    elif args.command == "demo":
        return handle_demo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
