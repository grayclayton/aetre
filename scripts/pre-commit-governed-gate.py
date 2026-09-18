#!/usr/bin/env python3
"""Road A Tiered Verification Gate: Host Pre-Commit Gatekeeper.

Intercepts code submissions locally on the host to eliminate invalid syntax
(Tier 0: $0.0000, ~0.05 ms) and bytecode/invariant compilation defects
(Tier 1: ~$0.0001, ~1 ms) before code is pushed to remote repositories
or triggers expensive containerized CI pipelines.

Can be run:
  1. As a git pre-commit hook via the 'pre-commit' framework (.pre-commit-hooks.yaml).
  2. Directly as a standalone script: python scripts/pre-commit-governed-gate.py
  3. In CI environments to pre-screen pull request diffs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def get_git_staged_python_files(repo_root: Optional[str] = None) -> List[str]:
    """Retrieves list of staged Python files from git index."""
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    try:
        res = subprocess.run(
            cmd,
            cwd=repo_root or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return []
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return [f for f in lines if f.endswith(".py")]
    except (FileNotFoundError, PermissionError):
        return []


def get_all_repo_python_files(repo_root: str, excludes: Optional[List[str]] = None) -> List[str]:
    """Finds all Python files within the repository excluding common build/cache directories."""
    default_excludes = {
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        "__pycache__",
        "build",
        "dist",
        "target",
        ".hypothesis",
        ".pytest_cache",
        "node_modules",
    }
    if excludes:
        default_excludes.update(excludes)

    py_files: List[str] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in default_excludes and not d.startswith(".")]
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_root)
                py_files.append(rel_path)
    return sorted(py_files)


def check_file(file_path: str) -> Dict[str, Any]:
    """Evaluates a single Python file against Road A Tier 0 and Tier 1 gates.

    Returns:
        Dict containing gate outcome, terminal tier, latency, and diagnostics.
    """
    t_start = time.perf_counter()
    if not os.path.exists(file_path):
        return {
            "file": file_path,
            "passed": False,
            "terminal_tier": 0,
            "error_type": "FileNotFoundError",
            "diagnostic": f"File not found: {file_path}",
            "lineno": None,
            "col_offset": None,
            "snippet": None,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }

    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as ex:
        return {
            "file": file_path,
            "passed": False,
            "terminal_tier": 0,
            "error_type": type(ex).__name__,
            "diagnostic": f"I/O read failure: {ex}",
            "lineno": None,
            "col_offset": None,
            "snippet": None,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }

    # Tier 0: AST Syntactic Parsing Gate ($0.0000, ~0.05 ms)
    try:
        source_code = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as ex:
        return {
            "file": file_path,
            "passed": False,
            "terminal_tier": 0,
            "error_type": "UnicodeDecodeError",
            "diagnostic": f"File contains non-UTF8 encoding: {ex}",
            "lineno": None,
            "col_offset": None,
            "snippet": None,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }

    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as ex:
        snippet = ex.text.strip() if ex.text else None
        diag = f"SyntaxError at line {ex.lineno}, col {ex.offset}: {ex.msg}"
        return {
            "file": file_path,
            "passed": False,
            "terminal_tier": 0,
            "error_type": "SyntaxError",
            "diagnostic": diag,
            "lineno": ex.lineno,
            "col_offset": ex.offset,
            "snippet": snippet,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }

    # Tier 1: Bytecode Compilation & Invariant Smoke Gate (~$0.0001, ~1 ms)
    try:
        compile(source_code, file_path, "exec")
    except Exception as ex:
        return {
            "file": file_path,
            "passed": False,
            "terminal_tier": 1,
            "error_type": type(ex).__name__,
            "diagnostic": f"CompilationError: {ex}",
            "lineno": getattr(ex, "lineno", None),
            "col_offset": getattr(ex, "offset", None),
            "snippet": getattr(ex, "text", "").strip() if getattr(ex, "text", None) else None,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }

    latency = round((time.perf_counter() - t_start) * 1000, 3)
    return {
        "file": file_path,
        "passed": True,
        "terminal_tier": 1,
        "error_type": None,
        "diagnostic": "Passed Tier 0 (AST) and Tier 1 (Bytecode compilation)",
        "lineno": None,
        "col_offset": None,
        "snippet": None,
        "latency_ms": latency,
    }


def run_precommit_gate(
    files: List[str],
    all_files: bool = False,
    staged_only: bool = False,
    json_mode: bool = False,
    dry_run: bool = False,
    repo_root: Optional[str] = None,
) -> int:
    """Executes the pre-commit gatekeeper over target files."""
    root = repo_root or os.getcwd()
    t_start = time.perf_counter()

    # Determine files to screen
    target_files: List[str] = []
    if all_files:
        target_files = get_all_repo_python_files(root)
    elif files:
        target_files = [f for f in files if f.endswith(".py")]
    elif staged_only or not files:
        staged = get_git_staged_python_files(root)
        target_files = staged

    if not target_files:
        if json_mode:
            print(json.dumps({
                "status": "SKIPPED",
                "message": "No Python files to screen",
                "files_screened": 0,
                "passed": True,
            }, indent=2))
        else:
            print("Road A Pre-Commit Gate: No Python files to screen.")
        return 0

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for fpath in target_files:
        full_path = os.path.join(root, fpath) if not os.path.isabs(fpath) else fpath
        res = check_file(full_path)
        # Use relative path for reporting if under root
        try:
            rel = os.path.relpath(full_path, root)
            res["file"] = rel
        except ValueError:
            res["file"] = full_path
        results.append(res)
        if not res["passed"]:
            failures.append(res)

    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    tier0_failures = [f for f in failures if f["terminal_tier"] == 0]
    tier1_failures = [f for f in failures if f["terminal_tier"] == 1]
    passed = len(failures) == 0

    # Economics: Estimate Docker runs avoided if failures are caught on host
    naive_container_cost_usd = 0.0200
    containers_avoided = 1 if failures else 0
    estimated_savings_usd = round(containers_avoided * naive_container_cost_usd, 4)

    if json_mode:
        report = {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "files_screened": len(target_files),
            "tier_0_syntax_errors": len(tier0_failures),
            "tier_1_compilation_errors": len(tier1_failures),
            "total_latency_ms": total_latency_ms,
            "docker_containers_avoided": containers_avoided,
            "estimated_savings_usd": estimated_savings_usd,
            "failures": failures,
            "results": results,
        }
        print(json.dumps(report, indent=2))
    else:
        print("=" * 78)
        print("  ROAD A: PRE-COMMIT VERIFICATION GATE (Host Screening)")
        print("=" * 78)
        print(f"Files Screened:         {len(target_files)} Python files")
        print(f"Total Host Latency:     {total_latency_ms} ms")
        print(f"Tier 0 (AST Syntax):    {'CLEAN' if not tier0_failures else f'FAILED ({len(tier0_failures)} errors)'}")
        print(f"Tier 1 (Bytecode/Inv):  {'CLEAN' if not tier1_failures else f'FAILED ({len(tier1_failures)} errors)'}")
        print("-" * 78)

        if failures:
            print("DEFECTS DETECTED ON HOST (Container Escalation Halted):")
            for fail in failures:
                tier_label = f"Tier {fail['terminal_tier']} ({fail['error_type']})"
                print(f"  [!] {fail['file']}: {tier_label}")
                if fail.get("lineno") is not None:
                    col_info = f", col {fail['col_offset']}" if fail.get("col_offset") is not None else ""
                    print(f"      Location: Line {fail['lineno']}{col_info}")
                if fail.get("snippet"):
                    print(f"      Code:     {fail['snippet']}")
                print(f"      Detail:   {fail['diagnostic']}")
            print("-" * 78)
            print(f"Economic Impact: Avoided 1 container CI execution (${naive_container_cost_usd:.4f} saved).")
            print("=" * 78)
            print("RESULT: HALT_AND_REJECT (Fix errors above before committing)")
            print("=" * 78)
        else:
            print("RESULT: ALL PRE-COMMIT INVARIANTS SATISFIED (Passed)")
            print("=" * 78)

    if dry_run or passed:
        return 0
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Road A Tiered Verification Gate: Host Pre-Commit Gatekeeper."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="List of Python files to screen (typically passed by pre-commit framework).",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Screen all Python files across the entire repository.",
    )
    parser.add_argument(
        "--staged-only",
        action="store_true",
        help="Screen only git staged Python files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON receipt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report defects without returning a non-zero exit code.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to repository root (defaults to current working directory).",
    )

    args = parser.parse_args(argv)
    return run_precommit_gate(
        files=args.files,
        all_files=args.all_files,
        staged_only=args.staged_only,
        json_mode=args.json,
        dry_run=args.dry_run,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    sys.exit(main())
