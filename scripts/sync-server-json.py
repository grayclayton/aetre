#!/usr/bin/env python3
"""Point server.json at the crate version.

The registry entry names a version and an image tag. Both were written by hand,
which is the same drift that once had serverInfo reporting 0.1.0 from a 0.2.0
release. Run this before publishing, or let the workflow run it.

Exits non-zero with --check if the file is out of date, so it can gate rather
than rewrite.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "crates" / "aetre-mcp" / "Cargo.toml"
SERVER_JSON = ROOT / "server.json"


def crate_version() -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if match:
            return match.group(1)
    raise SystemExit(f"no version found in {MANIFEST}")


def main() -> int:
    check_only = "--check" in sys.argv
    version = crate_version()
    doc = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    before = json.dumps(doc, indent=2, sort_keys=True)

    doc["version"] = version
    for package in doc.get("packages", []):
        if package["registryType"] == "cargo":
            package["version"] = version
        elif package["registryType"] == "oci":
            # The tag is the version for an OCI package; there is no version field.
            package["identifier"] = package["identifier"].rsplit(":", 1)[0] + ":" + version

    after = json.dumps(doc, indent=2, sort_keys=True)
    if before == after:
        print(f"server.json already matches the crate at {version}")
        return 0
    if check_only:
        print(f"server.json is out of date: the crate is at {version}", file=sys.stderr)
        return 1
    SERVER_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"server.json updated to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
