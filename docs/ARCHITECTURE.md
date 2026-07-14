# Architecture

## Pipeline Overview

This benchmark suite is built on a decoupled, multi-stage architecture. Each step operates via a fixed I/O contract, reading configurations from `configs/` and writing outputs to `models/` or `results/`.

1. **Model Acquisition**: Hugging Face Hub (FP16 Base).
2. **Quantization Engine**: 
   - AutoGPTQ & AutoAWQ for GPU-bound inferences.
   - llama.cpp (GGUF) for CPU-bound infrastructure.
3. **Serving Layer**: 
   - `vLLM` and `llama-server` wrapped in Docker containers. Both expose an identical `http://localhost:8000/v1/completions` API.
4. **Testing Harness**: Async Python client (`aiohttp`) sweeping across concurrency levels [1, 4, 8, 16, 32].
5. **Aggregation**: Automated metric consolidation into CSV and visual charts.

## Design Decisions

### 1. Cost-Aware Infra Split
We made the conscious decision to allow GGUF processing completely locally on a standard CPU workstation, reserving expensive rented cloud GPUs (e.g., RunPod, Colab) strictly for GPTQ/AWQ calibration phases and vLLM serving.

### 2. Standardized OpenAI API
By forcing both vLLM and llama.cpp to expose the same OpenAI-compatible REST API, we completely decouple the testing harness from the runtime. The load tester (`07_loadtest.py`) doesn't care if it's talking to C++ or Python.

### 3. Immutable Results
Every test run creates a timestamped JSON file with a unique UUID in `results/raw/`. We never overwrite past runs, making our benchmarking data perfectly auditable.
