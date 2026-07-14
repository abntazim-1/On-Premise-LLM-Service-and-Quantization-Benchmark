# Runbook: End-to-End Execution

Follow these steps to reproduce the benchmarks on a fresh machine.

## Prerequisites
- Linux / WSL2
- Python 3.10+
- At least 1 NVIDIA GPU with 16GB+ VRAM (for GPTQ/AWQ and vLLM)
- 32GB System RAM

## Step 1: Environment
```bash
python -m venv venv
source venv/bin/activate
make install
```

## Step 2: Download Model
```bash
make download
```
*Wait for the base model download to complete. It will save to `models/base_model`.*

## Step 3: Quantization
**GGUF (CPU Only):**
```bash
make quantize-gguf
```

**GPTQ & AWQ (Requires GPU):**
```bash
make quantize-gptq
make quantize-awq
```

## Step 4: Serving & Load Testing

You must start a server in a separate terminal before running a load test.

**Testing vLLM (GPTQ as example):**
```bash
# Terminal 1
./scripts/05_serve_vllm.sh gptq 8000

# Terminal 2
python scripts/07_loadtest.py --variant gptq --runtime vllm
```

**Testing llama.cpp (GGUF as example):**
```bash
# Terminal 1
./scripts/06_serve_llamacpp.sh q4_k_m 8000

# Terminal 2
python scripts/07_loadtest.py --variant q4_k_m --runtime llamacpp
```

## Step 5: Evaluation & Aggregation
Once all load tests are done, evaluate quality and aggregate:
```bash
python scripts/08_eval_quality.py --variant base
python scripts/08_eval_quality.py --variant gptq
# ... repeat for variants

make aggregate
```
Check `results/charts/` and `results/aggregated/results.csv` for your visual and tabular data.
