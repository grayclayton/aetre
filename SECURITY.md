# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or exposed
credential. Report it privately to `privacy@lithiumeel.com` with the affected
version, reproduction steps, and potential impact. Do not include real license
keys, private submissions, reviewer identities, or other sensitive data.

You should receive an acknowledgement within five business days. No bounty or
safe-harbor program is promised by this repository.

## Deployment boundary

The MCP stdio interface and embedded HTTP server process submitted text locally.
The HTTP server binds to loopback by default. Setting `AETRE_BIND_ADDRESS` to a
public interface exposes the API and should only be done behind an authenticated
reverse proxy with request limits and TLS. Non-loopback mode also requires an
`AETRE_HTTP_SERVER_TOKEN`; POST clients send it in `X-AETRE-Server-Token`.

Evaluation fingerprints are reproducibility identifiers, not signatures or
proof that a submission was accepted, externally reviewed, or registered by a
third party. The bundled server intentionally returns `404 Not Found` for
receipt-verification routes until a signed registry is implemented.
