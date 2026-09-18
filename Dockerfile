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

# Stage 2: Distroless minimal runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --uid 10001 --create-home aetre

WORKDIR /app

# Copy compiled binary from builder
COPY --from=builder /usr/src/aetre/target/release/aetre-mcp /app/aetre-mcp
RUN cp /app/aetre-mcp /usr/local/bin/aetre-mcp && chmod +x /app/aetre-mcp /usr/local/bin/aetre-mcp

# HTTP JSON-RPC API (/api/status, /api/tool). This is not an MCP transport:
# MCP clients speak stdio to this same binary.
EXPOSE 8080

ENV RUST_LOG=info
ENV PORT=8080
ENV AETRE_BIND_ADDRESS=0.0.0.0

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

USER aetre

ENTRYPOINT ["/app/aetre-mcp"]

