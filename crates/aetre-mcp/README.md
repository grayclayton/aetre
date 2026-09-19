# aetre-mcp

Model Context Protocol server for AETRE: decision-theoretic triage of candidate
work, and fail-closed gating of what an autonomous agent may execute.

32 tools over stdio, in two layers:

- **22 `aetre_*` tools** — value-of-information scoring, Kingman queue
  governance, quadratic staking, heavy-tailed evaluation, calibration and
  backtesting.
- **10 `governed_*` tools** — stopping policies, review boundaries, knapsack
  admission, invariant checks, shadow prices and signed decision receipts.

## Install

```bash
cargo install aetre-mcp
```

Point an MCP client at the binary:

```json
{ "mcpServers": { "aetre": { "command": "aetre-mcp" } } }
```

Every tool's schema costs context on connection, so a client that needs one
layer can be given one:

```bash
aetre-mcp --layer=micro   # the 10 Governed Agent tools
aetre-mcp --layer=macro   # the 22 AETRE tools
```

There is also a JSON-RPC HTTP mode for scripted access. It binds loopback by
default and refuses any other address without `AETRE_HTTP_SERVER_TOKEN`.

## Licence

AGPL-3.0-or-later. Every tool runs at every tier; a commercial licence covers
exemption from copyleft for closed-source and hosted use, not access to tools.

## Links

- Repository: <https://github.com/grayclayton/aetre>
- Website: <https://www.lithiumeel.com/aetre>
- mcp-name: io.github.grayclayton/aetre-mcp
