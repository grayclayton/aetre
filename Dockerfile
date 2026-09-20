# ==============================================================================
# AETRE production Dockerfile
# ==============================================================================

# Stage 1: Build the Rust release binary
FROM rust:1.80-slim-bookworm AS builder

WORKDIR /usr/src/aetre

# Copy manifests and sources, then build the locked MCP release binary.
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build --locked --release -p aetre-mcp

# Stage 2: distroless runtime.
#
# The binary is 1.6 MB; on debian:bookworm-slim the image was 32.7 MB, nearly
# all of it a userland nothing here uses. distroless/cc carries glibc, libgcc
# and ca-certificates and nothing else: no shell, no package manager.
#
# cc rather than static because this is the gnu target and needs glibc, and
# :nonroot because there is no useradd without a shell.
FROM gcr.io/distroless/cc-debian12:nonroot

WORKDIR /app

# COPY preserves the executable bit, which matters: there is no chmod here.
COPY --from=builder /usr/src/aetre/target/release/aetre-mcp /app/aetre-mcp

# HTTP JSON-RPC API (/api/status, /api/tool). This is not an MCP transport:
# MCP clients speak stdio to this same binary.
EXPOSE 8080

# The MCP Registry verifies ownership of an OCI image through this label,
# which must match the name in server.json exactly.
LABEL io.modelcontextprotocol.server.name="io.github.grayclayton/aetre-mcp"

ENV RUST_LOG=info
ENV AETRE_BIND_ADDRESS=0.0.0.0

# PORT is deliberately not set. Setting it puts the binary in HTTP mode, which
# would make `docker run -i` try to serve HTTP instead of answering MCP cleanly
# over stdio. Nothing is lost: AETRE_HTTP_SERVER_TOKEN alone enables HTTP, and
# the port already defaults to 8080.

# AETRE_HTTP_SERVER_TOKEN is REQUIRED at runtime and is deliberately not set here:
# a token baked into the image would be a published credential. The server refuses
# to bind a non-loopback address without one, and AETRE_BIND_ADDRESS above is
# 0.0.0.0, so without it the container logs a loud ERROR and serves no HTTP
# (it still answers MCP over stdio, which is what `docker run -i` uses).
#
#   docker run -e AETRE_HTTP_SERVER_TOKEN="$(openssl rand -hex 32)" ...
#   fly secrets set AETRE_HTTP_SERVER_TOKEN="$(openssl rand -hex 32)"   # before first deploy
#
# Authenticate POST /api/tool with the header:  X-Aetre-Server-Token: <token>


ENTRYPOINT ["/app/aetre-mcp"]

