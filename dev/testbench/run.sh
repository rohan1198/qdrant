#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QDRANT_URL="http://localhost:6334"

wait_for_qdrant() {
    echo "Waiting for Qdrant to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "$QDRANT_URL/healthz" > /dev/null 2>&1; then
            echo "Qdrant is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Qdrant did not start within 120 seconds."
    exit 1
}

DATASET="${2:-bgc}"

case "${1:-all}" in
    all)
        echo "=== Building Qdrant with hyperbolic feature ==="
        docker compose up -d --build
        wait_for_qdrant

        echo "=== Downloading dataset ($DATASET) ==="
        python3 download_dataset.py

        echo "=== Embedding & uploading ($DATASET, unified) ==="
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset "$DATASET" --cleanup-legacy

        echo "=== Running benchmarks ($DATASET) ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset "$DATASET"
        ;;
    hwv)
        wait_for_qdrant

        echo "=== Preparing HWV dataset ==="
        python3 ../datasets/hwv/prepare.py

        echo "=== Embedding & uploading (hwv, unified) ==="
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset hwv --cleanup-legacy

        echo "=== Running benchmarks (hwv) ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset hwv
        ;;
    benchmark)
        wait_for_qdrant
        echo "=== Running benchmarks ($DATASET) ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset "$DATASET"
        ;;
    embed)
        wait_for_qdrant
        echo "=== Embedding & uploading ($DATASET, unified) ==="
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset "$DATASET" --cleanup-legacy
        echo "=== Running benchmarks ($DATASET) ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset "$DATASET"
        ;;
    rebuild)
        echo "=== Rebuilding Qdrant binary ==="
        docker compose up -d --build
        wait_for_qdrant
        echo "Qdrant rebuilt and ready."
        ;;
    down)
        docker compose down -v
        ;;
    *)
        echo "Usage: ./run.sh [all|hwv|benchmark|embed|rebuild|down] [dataset]"
        echo "  dataset: bgc (default), hwv, wos, eurlex"
        exit 1
        ;;
esac

echo "=== Done ==="
