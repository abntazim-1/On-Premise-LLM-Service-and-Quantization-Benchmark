#!/bin/bash
# Script to serve models via vLLM with OpenAI-compatible API
# Usage: ./05_serve_vllm.sh [base|gptq|awq] [port]

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

echo "Starting vLLM server for $VARIANT model on port $PORT..."
echo "Command: python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH $QUANT_FLAG --port $PORT"

# Run vLLM's OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    $QUANT_FLAG \
    --port "$PORT" \
    --host 0.0.0.0
