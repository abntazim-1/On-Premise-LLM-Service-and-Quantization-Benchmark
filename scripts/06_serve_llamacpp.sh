#!/bin/bash
# Script to serve GGUF models via llama.cpp with OpenAI-compatible API
# Usage: ./06_serve_llamacpp.sh [q4_k_m|q5_k_m|q8_0] [port]

cd "$(dirname "$0")/.."

LEVEL=${1:-q4_k_m}
PORT=${2:-8000}

# Convert level to uppercase
LEVEL_UPPER=$(echo "$LEVEL" | tr '[:lower:]' '[:upper:]')
MODEL_PATH="models/gguf/model-${LEVEL_UPPER}.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    exit 1
fi

echo "Building llama.cpp docker image..."
docker build -t llamacpp-server -f docker/Dockerfile.llamacpp .

# Clean up any existing benchmark server container
echo "Stopping any existing benchmark server container..."
docker stop llm-benchmark-server 2>/dev/null || true
docker rm llm-benchmark-server 2>/dev/null || true

echo "Starting llama.cpp Docker container for GGUF $LEVEL_UPPER on port $PORT..."
docker run -d --name llm-benchmark-server \
    -v "$(pwd)/models:/app/models" \
    -p "$PORT:8000" \
    llamacpp-server \
    -m "/app/$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    -c 2048

echo "Waiting for llama.cpp server to start and load the GGUF model (timeout: 60s)..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    # Check if container is running
    if ! docker ps -q --filter name=llm-benchmark-server > /dev/null; then
        echo "Error: Container crashed during startup."
        docker logs llm-benchmark-server
        exit 1
    fi

    # Query OpenAI compatibility models endpoint
    if curl -s http://localhost:$PORT/v1/models > /dev/null; then
        echo "llama.cpp server is up and running on port $PORT!"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "Waiting... (${elapsed}s elapsed)"
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Server failed to respond within $timeout seconds."
    docker logs llm-benchmark-server
    docker stop llm-benchmark-server
    exit 1
fi



