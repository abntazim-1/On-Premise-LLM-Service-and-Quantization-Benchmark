# On-Premise LLM Serving & Quantization Benchmark

An enterprise-grade, reproducible benchmark suite designed to evaluate the throughput, latency, memory footprint, and quality tradeoffs of running open-source LLMs (like Llama-2-7B) on-premise.

This framework explicitly quantizes a base FP16 model using **GPTQ**, **AWQ**, and **GGUF** formats, and serves them via **vLLM** (GPU) and **llama.cpp** (CPU) behind an OpenAI-compatible API to perform apples-to-apples load testing.

## Quick Results Snapshot
*(Placeholder: Run `make bench-all` to generate your local metrics)*

| Variant | Runtime | P50 Latency | Throughput (tok/s) | Peak VRAM |
|---------|---------|-------------|--------------------|-----------|
| FP16    | vLLM    | -           | -                  | -         |
| GPTQ-4B | vLLM    | -           | -                  | -         |
| AWQ-4B  | vLLM    | -           | -                  | -         |
| GGUF-Q4 | llama   | -           | -                  | -         |

Read the full performance write-up in [docs/RESULTS.md](docs/RESULTS.md).

## Getting Started

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for detailed step-by-step instructions.

```bash
# 1. Install dependencies
make install

# 2. Download the base model
make download

# 3. Quantize (Requires GPU for GPTQ/AWQ)
make quantize-gptq
make quantize-awq
make quantize-gguf

# 4. Run load tests & evaluations
# (Start the respective server via scripts/05_serve_vllm.sh first)
make loadtest
make eval
make aggregate
```

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design decisions, including the CPU/GPU compute split strategy.
