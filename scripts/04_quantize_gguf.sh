#!/bin/bash
# Exit on any error
set -e

echo "=== GGUF Quantization Pipeline ==="

MODEL_DIR="models/base_model"
GGUF_DIR="models/gguf"
mkdir -p "$GGUF_DIR"

if ! command -v make &> /dev/null || ! command -v g++ &> /dev/null; then
    echo "Error: 'make' and 'g++' are required to build llama.cpp."
    echo "Please install build-essential (e.g., 'sudo apt install build-essential') and try again."
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Base model not found at $MODEL_DIR. Run the download script first."
    exit 1
fi

# Step 1: Ensure llama.cpp is available for the conversion and quantization tools
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    # Build the quantization tool
    echo "Building llama.cpp..."
    make -j
    # Install Python dependencies for the conversion script
    pip install -r requirements.txt
    cd ..
else
    echo "llama.cpp repository already exists."
fi

# Step 2: Convert base Hugging Face model to unquantized (FP16) GGUF format
FP16_GGUF="$GGUF_DIR/model-fp16.gguf"
if [ ! -f "$FP16_GGUF" ]; then
    echo "Converting $MODEL_DIR to FP16 GGUF..."
    python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$FP16_GGUF" --outtype f16
else
    echo "FP16 GGUF already exists at $FP16_GGUF. Skipping conversion."
fi

# Step 3: Quantize to specific levels (Q4_K_M, Q5_K_M, Q8_0)
# We read levels from config or hardcode the expected ones from configs/quant_gguf.yaml
LEVELS=("q4_k_m" "q5_k_m" "q8_0")

echo "Quantizing to target levels..."
for LEVEL in "${LEVELS[@]}"; do
    # Convert level string to uppercase as expected by llama-quantize
    LEVEL_UPPER=$(echo "$LEVEL" | tr '[:lower:]' '[:upper:]')
    OUT_PATH="$GGUF_DIR/model-${LEVEL}.gguf"
    
    if [ ! -f "$OUT_PATH" ]; then
        echo "--> Quantizing to $LEVEL_UPPER..."
        ./llama.cpp/llama-quantize "$FP16_GGUF" "$OUT_PATH" "$LEVEL_UPPER"
    else
        echo "--> Quantized model $OUT_PATH already exists. Skipping."
    fi
done

echo "=== GGUF Quantization Complete ==="
