FROM rust:bookworm AS chef
# We only pay the installation cost once,
# it will be cached from the second build onwards
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG EXTRA_FEATURES=""

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY rust-toolchain.toml .
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --features="$EXTRA_FEATURES"
# Build application
COPY . .

RUN cargo build --release -p bayesian-regression-101-core --features="$EXTRA_FEATURES"


# We do not need the Rust toolchain to run the binary!
FROM debian:bookworm-slim AS base-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# add user with /app as home directory 
RUN groupadd -g 1000 bayesian-regression-101 \
    && useradd -u 1000 -g bayesian-regression-101 -s /bin/sh -m bayesian-regression-101 \
    && mkdir /app \
    && chown bayesian-regression-101:bayesian-regression-101 /app

WORKDIR /app

FROM base-runtime AS bayesian-regression-101-core

COPY --from=builder /app/target/release/bayesian-regression-101-core /usr/local/bin

USER 1000:0

ENTRYPOINT ["/usr/local/bin/bayesian-regression-101-core"]
