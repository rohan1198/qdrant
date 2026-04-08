# Stage 1: Build Qdrant with hyperbolic feature
FROM rust:1.94-bookworm AS builder

RUN apt-get update && apt-get install -y \
    clang lld cmake protobuf-compiler pkg-config \
    libunwind-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qdrant
COPY . .

RUN RUSTFLAGS="-C link-arg=-fuse-ld=lld" \
    cargo build --release --features hyperbolic --bin qdrant

# Stage 2: Runtime
FROM debian:13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /qdrant/target/release/qdrant /qdrant/qdrant
COPY --from=builder /qdrant/config /qdrant/config
COPY --from=builder /qdrant/tools/entrypoint.sh /qdrant/entrypoint.sh

WORKDIR /qdrant

ENV TZ=Etc/UTC RUN_MODE=production

EXPOSE 6333 6334

CMD ["./qdrant"]
