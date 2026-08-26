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

# Expose standard MCP SSE port
EXPOSE 8080

ENV RUST_LOG=info
ENV PORT=8080
ENV AETRE_BIND_ADDRESS=0.0.0.0

USER aetre

ENTRYPOINT ["/app/aetre-mcp", "--serve", "--headless"]

