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

case "${1:-all}" in
    all)
        echo "=== Building Qdrant with hyperbolic feature ==="
        docker compose up -d --build
        wait_for_qdrant

        echo "=== Downloading dataset ==="
        python3 download_dataset.py

        echo "=== Embedding & uploading ==="
        python3 embed.py --qdrant-url "$QDRANT_URL"

        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    benchmark)
        wait_for_qdrant
        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    embed)
        wait_for_qdrant
        echo "=== Embedding & uploading ==="
        python3 embed.py --qdrant-url "$QDRANT_URL"
        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    down)
        docker compose down -v
        ;;
    *)
        echo "Usage: ./run.sh [all|benchmark|embed|down]"
        exit 1
        ;;
esac

echo "=== Done ==="
