#!/usr/bin/env python3
"""Hold the Rust Tier 0 gate to CPython's verdict.

`governed_gate_pr` reimplements, in Rust, a judgement CPython already makes
authoritatively with `ast.parse`. Nothing compared the two, and the Rust copy
drifted: it had no notion of a comment, so an apostrophe or an unmatched bracket
in ordinary English prose was read as an open string or an unbalanced paren. It
rejected 7 of the 75 real Python files across this repo and two private ones.

The two cannot be made identical - the gate is a shallow balance scan that runs
before anything expensive, not a parser - so this asserts the asymmetry that
actually matters:

  * NEVER reject code CPython accepts. A false positive halts good work before
    verification ever runs, which is the failure this gate exists to avoid.
  * Catch the obvious breakage. Misses are permitted by design and fall through
    to real verification, but the catch rate must not silently collapse to zero.

Usage:
    python3 scripts/check-gate-against-cpython.py [--bin path/to/aetre-mcp]
"""
import argparse
import ast
import json
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("/target/", "/.venv/", "/node_modules/", "__pycache__")

# Deterministic seed: a flaky gate check would train people to ignore it.
MUTATION_SEED = 20260920
MUTATIONS_PER_FILE = 12
MAX_MUTANTS = 150
MIN_CATCH_RATE = 0.80

# Cases the gate must always get right, independent of what is in the repo.
# The first three are the regression this check was written for.
FIXED_CASES = [
    ("x = 1\n# under this batch's shadow price\ny = 2\n", True),
    ("x = 1\n# don't do this\ny = 2\n", True),
    ("x = 1\n# see foo( for details\ny = 2\n", True),
    ('def f():\n    """a ( paren in prose"""\n    return 1\n', True),
    ('def f():\n    """he said "hi" to me"""\n    return 1\n', True),
    ("x = \"# not a comment\"\ny = 2\n", True),
    ("x = 'it\\'s fine'\ny = 2\n", True),
    ("", True),
    ("def broken(:\n    pass\n", False),
    ("x = (1 + 2))\n", False),
    ("x = 'unterminated\n", False),
    ("d = {'a': 1\n", False),
    ("xs = [1, 2\n", False),
]


def gate(binary, sources):
    """Returns the gate's `passed` verdict for each source, in order."""
    init = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "gate-check", "version": "1"}}}
    msgs = [init] + [
        {"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
         "params": {"name": "governed_gate_pr", "arguments": {"code": src}}}
        for i, src in enumerate(sources)
    ]
    payload = "\n".join(json.dumps(m) for m in msgs) + "\n"
    proc = subprocess.run([binary], input=payload.encode("utf-8"),
                          capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"the server exited {proc.returncode}: "
                 f"{proc.stderr.decode('utf-8', 'replace')[:400]}")
    seen = {}
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("id"):
            seen[msg["id"]] = msg
    verdicts = []
    for i in range(len(sources)):
        msg = seen.get(i + 1)
        if msg is None:
            sys.exit(f"no response for source {i}; the server answered "
                     f"{len(seen)} of {len(sources)} calls")
        body = json.loads(msg["result"]["content"][0]["text"])
        verdicts.append((body.get("passed"), body.get("diagnostic", "")))
    return verdicts


def cpython_accepts(source):
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False
    except ValueError:
        # Source containing a null byte; CPython rejects it too.
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=str(ROOT / "target" / "release" / "aetre-mcp"))
    args = parser.parse_args()
    binary = args.bin
    if not pathlib.Path(binary).exists() and pathlib.Path(binary + ".exe").exists():
        binary += ".exe"

    failures = []

    # --- the repository's own Python, which must never be rejected ----------
    files = [p for p in ROOT.rglob("*.py")
             if not any(s in p.as_posix() for s in SKIP)]
    sources, valid = [], []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if cpython_accepts(text):
            sources.append(text)
            valid.append(path)
    verdicts = gate(binary, sources)
    rejected = [(p, d) for (p, _), (ok, d) in zip(zip(valid, sources), verdicts)
                if not ok]
    print(f"repository corpus: {len(valid)} files CPython accepts")
    for path, why in rejected:
        failures.append(f"false positive: {path.relative_to(ROOT).as_posix()} - {why}")
    print(f"  wrongly rejected: {len(rejected)}")

    # --- fixed cases --------------------------------------------------------
    verdicts = gate(binary, [src for src, _ in FIXED_CASES])
    wrong = 0
    for (src, want), (got, why) in zip(FIXED_CASES, verdicts):
        if got != want:
            wrong += 1
            verb = "rejected" if want else "accepted"
            failures.append(f"fixed case wrongly {verb}: {src!r} - {why}")
    print(f"fixed cases: {len(FIXED_CASES) - wrong}/{len(FIXED_CASES)} correct")

    # --- mutants CPython rejects, which the gate should mostly catch --------
    rng = random.Random(MUTATION_SEED)
    mutants = []
    for text in sources:
        spots = [i for i, ch in enumerate(text) if ch in "()[]{}'\""]
        rng.shuffle(spots)
        for i in spots[:MUTATIONS_PER_FILE]:
            candidate = text[:i] + text[i + 1:]
            if not cpython_accepts(candidate):
                mutants.append(candidate)
            if len(mutants) >= MAX_MUTANTS:
                break
        if len(mutants) >= MAX_MUTANTS:
            break
    if mutants:
        verdicts = gate(binary, mutants)
        caught = sum(1 for ok, _ in verdicts if not ok)
        rate = caught / len(mutants)
        print(f"mutants CPython rejects: {caught}/{len(mutants)} caught ({rate:.0%})")
        if rate < MIN_CATCH_RATE:
            failures.append(
                f"catch rate {rate:.0%} is below the {MIN_CATCH_RATE:.0%} floor; "
                f"the gate has stopped rejecting broken code"
            )
    else:
        print("mutants: none generated")

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for line in failures[:20]:
            print(f"  {line}")
        return 1
    print("\nthe gate never contradicts CPython on valid code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
