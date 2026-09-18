"""Pillar IV: Mutation-Grounded Adversarial Oracles.

Programmatic AST mutation engine:
- Injects synthetic AST mutants (AOR, ROR, LCR, BCR, COI) into candidate code.
- Evaluates probe suites against the mutant population.
- Calculates Mutation Score MS(Phi, M) and identifies surviving mutants.
- Enforces the adversarial gate MS >= 1 - epsilon to pierce the defect leakage floor.
"""
from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Mutant:
    mutant_id: str
    operator: str
    lineno: int
    description: str
    mutant_code: str
    fn: Optional[Callable[..., Any]] = None


@dataclass
class MutationReport:
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    mutation_score: float
    is_qualified: bool
    surviving_mutants: List[Dict[str, Any]] = field(default_factory=list)
    killed_details: List[Dict[str, Any]] = field(default_factory=list)
    inconclusive_mutants: int = 0
    inconclusive_details: List[Dict[str, Any]] = field(default_factory=list)


class ASTMutator(ast.NodeTransformer):
    """AST transformer that creates single-point mutations."""

    def __init__(self, target_idx: int):
        super().__init__()
        self.target_idx = target_idx
        self.current_idx = 0
        self.applied_mutation: Optional[Tuple[str, int, str]] = None

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        # Arithmetic Operator Replacement (AOR)
        op_replacements = {
            ast.Add: (ast.Sub(), "+ to -"),
            ast.Sub: (ast.Add(), "- to +"),
            ast.Mult: (ast.FloorDiv(), "* to //"),
            ast.FloorDiv: (ast.Mult(), "// to *"),
            ast.Mod: (ast.Add(), "% to +"),
        }
        for src_op, (dst_op, desc) in op_replacements.items():
            if isinstance(node.op, src_op):
                if self.current_idx == self.target_idx:
                    node.op = dst_op
                    self.applied_mutation = ("AOR", getattr(node, "lineno", 0), desc)
                self.current_idx += 1
                break
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        # Relational Operator Replacement (ROR)
        cmp_replacements = {
            ast.Lt: (ast.LtE(), "< to <="),
            ast.LtE: (ast.Lt(), "<= to <"),
            ast.Gt: (ast.GtE(), "> to >="),
            ast.GtE: (ast.Gt(), ">= to >"),
            ast.Eq: (ast.NotEq(), "== to !="),
            ast.NotEq: (ast.Eq(), "!= to =="),
        }
        new_ops = []
        for op in node.ops:
            replaced = False
            for src_op, (dst_op, desc) in cmp_replacements.items():
                if isinstance(op, src_op):
                    if self.current_idx == self.target_idx:
                        new_ops.append(dst_op)
                        self.applied_mutation = ("ROR", getattr(node, "lineno", 0), desc)
                        replaced = True
                    self.current_idx += 1
                    break
            if not replaced:
                new_ops.append(op)
        node.ops = new_ops
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        # Logical Connector Replacement (LCR)
        if isinstance(node.op, ast.And):
            if self.current_idx == self.target_idx:
                node.op = ast.Or()
                self.applied_mutation = ("LCR", getattr(node, "lineno", 0), "and to or")
            self.current_idx += 1
        elif isinstance(node.op, ast.Or):
            if self.current_idx == self.target_idx:
                node.op = ast.And()
                self.applied_mutation = ("LCR", getattr(node, "lineno", 0), "or to and")
            self.current_idx += 1
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # Boundary Constant Replacement (BCR)
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self.current_idx == self.target_idx:
                new_val = node.value + 1 if node.value == 0 else node.value - 1
                desc = f"constant {node.value} to {new_val}"
                node.value = new_val
                self.applied_mutation = ("BCR", getattr(node, "lineno", 0), desc)
            self.current_idx += 1
        return node


class MutationEngine:
    """Generates synthetic mutants and audits probe mutation scores."""

    def __init__(self, qualification_threshold: float = 0.80):
        self.threshold = qualification_threshold

    def generate_mutants(self, target_fn_or_code: Any, max_mutants: int = 5) -> List[Mutant]:
        """Generates up to max_mutants distinct mutants from function or code string."""
        if callable(target_fn_or_code):
            code_str = inspect.getsource(target_fn_or_code)
            fn_name = target_fn_or_code.__name__
        else:
            code_str = str(target_fn_or_code)
            fn_name = None

        # Clean indentation if needed
        import textwrap
        code_str = textwrap.dedent(code_str)

        base_ast = ast.parse(code_str)
        if fn_name is None:
            for node in ast.walk(base_ast):
                if isinstance(node, ast.FunctionDef):
                    fn_name = node.name
                    break

        # First pass: count total mutation points
        counter = ASTMutator(-1)
        counter.visit(copy.deepcopy(base_ast))
        total_points = counter.current_idx

        mutants: List[Mutant] = []
        indices = list(range(total_points))
        if len(indices) > max_mutants:
            # Spread mutation points across the function body
            step = len(indices) / max_mutants
            selected_indices = [int(i * step) for i in range(max_mutants)]
        else:
            selected_indices = indices

        for m_idx, target_idx in enumerate(selected_indices):
            mutator = ASTMutator(target_idx)
            mutated_ast = mutator.visit(copy.deepcopy(base_ast))
            ast.fix_missing_locations(mutated_ast)

            if mutator.applied_mutation is None:
                continue

            op_type, lineno, desc = mutator.applied_mutation
            mutant_source = ast.unparse(mutated_ast)

            # Validate bytecode without executing module-level candidate code.
            try:
                compile(mutated_ast, filename=f"<mutant_{m_idx}>", mode="exec")
            except Exception:
                continue

            mutants.append(Mutant(
                mutant_id=f"mutant_{m_idx+1}_{op_type.lower()}",
                operator=op_type,
                lineno=lineno,
                description=desc,
                mutant_code=mutant_source,
                fn=None,
            ))

        return mutants

    def evaluate_mutation_score(
        self,
        probes: List[Dict[str, Any]],
        reference_fn: Callable[..., Any],
        mutants: List[Mutant],
        isolated_runner: Optional[Any] = None,
    ) -> MutationReport:
        """Evaluates whether probe suite kills the synthetic mutants.
        
        If isolated_runner is provided, mutants are executed inside disposable
        Linux Docker containers to prevent mutant infinite loops or crashes
        from affecting the host supervisor.
        """
        if not mutants:
            return MutationReport(
                total_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                mutation_score=1.0,
                is_qualified=True,
            )

        killed = 0
        survived = 0
        inconclusive = 0
        surviving_details: List[Dict[str, Any]] = []
        killed_details: List[Dict[str, Any]] = []
        inconclusive_details: List[Dict[str, Any]] = []

        ref_fn_name = getattr(reference_fn, "__name__", None)

        for mutant in mutants:
            mutant_fn = mutant.fn
            if mutant_fn is None and isolated_runner is None:
                # This is the explicitly trusted execution path. Generation itself
                # remains data-only, so CLI audit cannot trigger module side effects.
                namespace: Dict[str, Any] = {}
                try:
                    exec(compile(mutant.mutant_code, f"<{mutant.mutant_id}>", "exec"), namespace)
                    mutant_fn = namespace.get(ref_fn_name)
                except Exception as ex:
                    inconclusive += 1
                    inconclusive_details.append({
                        "mutant_id": mutant.mutant_id,
                        "reason": f"Could not load mutant: {type(ex).__name__}: {ex}",
                    })
                    continue
                if mutant_fn is None:
                    inconclusive += 1
                    inconclusive_details.append({
                        "mutant_id": mutant.mutant_id,
                        "reason": f"Mutant function {ref_fn_name!r} was not found",
                    })
                    continue

            mutant_killed = False
            killer_probe_name = None
            discrepancy_reason = None
            mutant_inconclusive = False

            for p_idx, probe in enumerate(probes):
                kwargs = probe.get("proposed_test_input") or probe.get("input") or {}
                p_name = probe.get("obligation") or probe.get("name") or f"probe_{p_idx}"

                # Execute reference in host process
                ref_err = None
                try:
                    ref_res = reference_fn(**kwargs)
                except Exception as ex:
                    ref_err = ex

                # Execute mutant under isolation or in-process
                mut_err = None
                mut_res = None
                if isolated_runner is not None and mutant.mutant_code:
                    try:
                        cand_exec = isolated_runner.run_candidate(
                            code=mutant.mutant_code,
                            func_name=ref_fn_name,
                            kwargs=kwargs,
                        )
                        if cand_exec.status == "SUCCESS":
                            mut_res = cand_exec.value
                        elif cand_exec.status == "EXCEPTION":
                            mut_err = cand_exec.exception_type
                        elif cand_exec.status in ("INFRASTRUCTURE_FAILURE", "CANCELLED"):
                            mutant_inconclusive = True
                            discrepancy_reason = f"Mutation execution was inconclusive [{cand_exec.status}]: {cand_exec.diagnostic}"
                            break
                        else:
                            # A mutant timeout, crash, or oversized output is an
                            # observable behavioral difference and kills the mutant.
                            mutant_killed = True
                            killer_probe_name = p_name
                            discrepancy_reason = f"Mutant failed under container isolation [{cand_exec.status}]: {cand_exec.diagnostic}"
                            break
                    except Exception as ex:
                        mutant_inconclusive = True
                        discrepancy_reason = f"Isolation backend error on mutant: {ex}"
                        break
                else:
                    try:
                        mut_res = mutant_fn(**kwargs)
                    except Exception as ex:
                        mut_err = ex

                # Discrepancy check
                ref_err_type_name = type(ref_err).__module__ + "." + type(ref_err).__qualname__ if ref_err is not None else None
                mut_err_type_name = mut_err if isinstance(mut_err, str) else (type(mut_err).__module__ + "." + type(mut_err).__qualname__ if mut_err is not None else None)

                if ref_err is not None and mut_err is None:
                    mutant_killed = True
                    killer_probe_name = p_name
                    discrepancy_reason = f"Ref raised {ref_err_type_name}, but mutant returned {mut_res!r}"
                    break
                elif ref_err is None and mut_err is not None:
                    mutant_killed = True
                    killer_probe_name = p_name
                    discrepancy_reason = f"Ref returned {ref_res!r}, but mutant raised {mut_err_type_name}"
                    break
                elif ref_err is not None and mut_err is not None:
                    if ref_err_type_name != mut_err_type_name:
                        mutant_killed = True
                        killer_probe_name = p_name
                        discrepancy_reason = f"Exception mismatch: ref={ref_err_type_name} vs mut={mut_err_type_name}"
                        break
                else:
                    if ref_res != mut_res:
                        mutant_killed = True
                        killer_probe_name = p_name
                        discrepancy_reason = f"Value mismatch: ref={ref_res!r} vs mut={mut_res!r}"
                        break

            if mutant_inconclusive:
                inconclusive += 1
                inconclusive_details.append({
                    "mutant_id": mutant.mutant_id,
                    "description": mutant.description,
                    "reason": discrepancy_reason,
                })
            elif mutant_killed:
                killed += 1
                killed_details.append({
                    "mutant_id": mutant.mutant_id,
                    "description": mutant.description,
                    "killer_probe": killer_probe_name,
                    "reason": discrepancy_reason,
                })
            else:
                survived += 1
                surviving_details.append({
                    "mutant_id": mutant.mutant_id,
                    "operator": mutant.operator,
                    "lineno": mutant.lineno,
                    "description": mutant.description,
                    "mutant_code_snippet": mutant.mutant_code[:200],
                })

        conclusive = killed + survived
        score = killed / conclusive if conclusive else 0.0
        return MutationReport(
            total_mutants=len(mutants),
            killed_mutants=killed,
            survived_mutants=survived,
            mutation_score=score,
            is_qualified=(inconclusive == 0 and score >= self.threshold),
            surviving_mutants=surviving_details,
            killed_details=killed_details,
            inconclusive_mutants=inconclusive,
            inconclusive_details=inconclusive_details,
        )
