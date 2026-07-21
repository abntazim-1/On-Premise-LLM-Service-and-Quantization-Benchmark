#!/bin/bash
# Script to serve models via vLLM with OpenAI-compatible API
# Usage: ./05_serve_vllm.sh [base|gptq|awq] [port]

cd "$(dirname "$0")/.."

VARIANT=${1:-base}
PORT=${2:-8000}

case $VARIANT in
  base)
    MODEL_PATH="models/base_model"
    QUANT_FLAG=""
    ;;
  gptq)
    MODEL_PATH="models/gptq_4bit"
    QUANT_FLAG="--quantization gptq"
    ;;
  awq)
    MODEL_PATH="models/awq_4bit"
    QUANT_FLAG="--quantization awq"
    ;;
  *)
    echo "Usage: $0 [base|gptq|awq] [port]"
    exit 1
    ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory $MODEL_PATH not found. Have you run the quantization step?"
    exit 1
fi

echo "Building vLLM docker image (if not exists)..."
docker build -t vllm-server -f docker/Dockerfile.vllm .

# Clean up any existing benchmark server container
echo "Stopping any existing benchmark server container..."
docker stop llm-benchmark-server 2>/dev/null || true
docker rm llm-benchmark-server 2>/dev/null || true

echo "Starting vLLM Docker container for $VARIANT model on port $PORT..."
docker run -d --name llm-benchmark-server --gpus all \
    -v "$(pwd)/models:/app/models" \
    -p "$PORT:8000" \
    vllm-server \
    --model "/app/$MODEL_PATH" $QUANT_FLAG \
    --port 8000 \
    --host 0.0.0.0

echo "Waiting for vLLM server to start and load the model (timeout: 180s)..."
timeout=180
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
        echo "vLLM server is up and running on port $PORT!"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "Waiting... (${elapsed}s elapsed)"
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Server failed to respond within $timeout seconds."
    docker logs llm-benchmark-server
    docker stop llm-benchmark-server
    exit 1
fi

