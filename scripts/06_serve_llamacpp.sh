#!/bin/bash
# Script to serve GGUF models via llama.cpp with OpenAI-compatible API
# Usage: ./06_serve_llamacpp.sh [q4_k_m|q5_k_m|q8_0] [port]

LEVEL=${1:-q4_k_m}
PORT=${2:-8000}

# Convert level to uppercase
LEVEL_UPPER=$(echo "$LEVEL" | tr '[:lower:]' '[:upper:]')
MODEL_PATH="models/gguf/model-${LEVEL_UPPER}.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    exit 1
fi

if [ ! -f "llama.cpp/llama-server" ]; then
    echo "llama-server executable not found. Attempting to build it..."
    cd llama.cpp
    make -j llama-server
    cd ..
fi

echo "Starting llama.cpp server for GGUF $LEVEL_UPPER on port $PORT..."
echo "Command: ./llama.cpp/llama-server -m $MODEL_PATH --port $PORT --host 0.0.0.0 -c 2048"

# Run llama.cpp's built-in server which is OpenAI API compatible
./llama.cpp/llama-server \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    -c 2048
