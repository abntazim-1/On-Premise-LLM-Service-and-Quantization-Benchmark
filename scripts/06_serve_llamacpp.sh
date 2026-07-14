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

echo "Starting llama.cpp Docker container for GGUF $LEVEL_UPPER on port $PORT..."
echo "Command: docker run --rm -v $(pwd)/models:/app/models -p $PORT:8000 llamacpp-server ..."

docker run --rm \
    -v "$(pwd)/models:/app/models" \
    -p "$PORT:8000" \
    llamacpp-server \
    -m "/app/$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    -c 2048


