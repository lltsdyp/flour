# Builds the flour engine plus the standalone distkv Master/Worker binaries.
# Build context is the repository root (see docker-compose.yml).
FROM rust:1-slim AS builder
WORKDIR /build

# System deps: tokenizers (onig) needs a C toolchain; reqwest needs TLS headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release \
    --bin flour --bin flour-master --bin flour-worker

FROM debian:bookworm-slim AS runtime
# curl is included so the cluster can be smoke-tested from inside the network
# (worker data-path hostnames only resolve within the compose network).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/flour /usr/local/bin/flour
COPY --from=builder /build/target/release/flour-master /usr/local/bin/flour-master
COPY --from=builder /build/target/release/flour-worker /usr/local/bin/flour-worker
