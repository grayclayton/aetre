#!/usr/bin/env python3
"""Keep the two copies of the lexical scorer from drifting apart.

`heuristics.rs` and the novelty reference it was measured on exist twice: once
here in `crates/aetre-mcp`, and once in the private engine's `crates/aetre-studio`.
They are meant to be the same scorer. They were not: the private copy never
received the lexicon fix, so it still counted "novel", "paradigm", "breakthrough"
and "frontier" as evidence of novelty, and a content-free abstract ranked in the
top 2.8% of real submissions there while ranking top 63.5% here. Nobody noticed,
because each copy only ever checked itself.

The in-crate fingerprint test catches an edit that forgets to regenerate the
reference. It cannot catch an edit made consistently in one repo and not the
other - both copies stay internally consistent while disagreeing with each other.
That is exactly what happened, and it is what this closes.

Usage:
    python3 scripts/check-scorer-sync.py --against ../epistemic_triage_engine
    AETRE_SYNC_AGAINST=../epistemic_triage_engine python3 scripts/check-scorer-sync.py
"""
import argparse
import difflib
import hashlib
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files that must be byte-identical between checkouts, by role. The crate
# directory differs between the two repos, so each is searched for.
SHARED = {
    "scorer": ["crates/aetre-mcp/src/heuristics.rs",
               "crates/aetre-studio/src/heuristics.rs"],
    "novelty reference": ["crates/aetre-core/reference/novelty_reference_v1.json"],
}


def locate(root, candidates, role):
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    sys.exit(f"no {role} found under {root}; looked for "
             + ", ".join(candidates))


def normalise(path):
    """Compare content, not line endings: the repos differ on CRLF."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--against", default=os.environ.get("AETRE_SYNC_AGAINST"),
                        help="root of the other checkout")
    parser.add_argument("--from", dest="here", default=str(ROOT),
                        help="root of this checkout (default: this repo)")
    args = parser.parse_args()
    if not args.against:
        sys.exit("--against <other checkout> is required "
                 "(or set AETRE_SYNC_AGAINST)")

    here = pathlib.Path(args.here).resolve()
    there = pathlib.Path(args.against).resolve()
    if not there.exists():
        sys.exit(f"the other checkout does not exist: {there}")

    drifted = []
    for role, candidates in SHARED.items():
        a = locate(here, candidates, role)
        b = locate(there, candidates, role)
        text_a, text_b = normalise(a), normalise(b)
        digest = hashlib.sha256(text_a.encode("utf-8")).hexdigest()[:12]
        if text_a == text_b:
            print(f"  {role}: identical ({digest})")
            continue
        drifted.append(role)
        print(f"  {role}: DRIFTED")
        print(f"    {a}")
        print(f"    {b}")
        diff = list(difflib.unified_diff(
            text_a.splitlines(), text_b.splitlines(),
            fromfile=str(a), tofile=str(b), lineterm="", n=1))
        for line in diff[:40]:
            print(f"      {line}")
        if len(diff) > 40:
            print(f"      ... {len(diff) - 40} more diff lines")

    if drifted:
        print(f"\nFAILED: {', '.join(drifted)} differ between the two checkouts.")
        print("A percentile is only valid for the scorer it was measured on, so a")
        print("scorer that differs between repos makes the shared reference wrong")
        print("in one of them. Copy the intended version across and regenerate:")
        print("  aetre-mcp --emit-novelty-reference <corpus.json> "
              "openreview-iclr-neurips \"<description>\"")
        return 1

    print("\nboth checkouts run the same scorer against the same reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
