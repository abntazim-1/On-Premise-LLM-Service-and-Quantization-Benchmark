# CV Bullets — On-Premise LLM Serving & Quantization Benchmark

> **Format:** Google XYZ — *"Accomplished [X] as measured by [Y], by doing [Z]"*

---

## 🏆 Primary Bullet (Headline / Top of Project Entry)

> Engineered a **reproducible, enterprise-grade LLM quantization and serving benchmark suite**, reducing model memory footprint by up to 60% with sub-1% perplexity degradation, by designing a fully decoupled multi-stage pipeline that quantizes a 7B parameter FP16 base model across GPTQ, AWQ, and GGUF formats and serves each variant behind standardized OpenAI-compatible Docker containers for apples-to-apples load testing.

---

## 📐 Supporting Bullets (Use 2–3 as sub-points or rotate by role)

### Quantization & Model Optimization

> Achieved 4-bit model compression across three industry-standard quantization formats (GPTQ, AWQ, GGUF Q4/Q5/Q8), enabling on-premise deployment on consumer-grade hardware with 16GB VRAM, by implementing calibrated quantization pipelines using AutoGPTQ and AutoAWQ with a 128-sample Wikitext-2 calibration set and validating quality retention via automated perplexity scoring on a held-out test split.

---

### Inference Infrastructure & Serving

> Eliminated runtime-specific client branching and reduced infrastructure coupling to zero by architecting a unified serving layer that wraps both vLLM (GPU, GPTQ/AWQ) and llama.cpp (CPU, GGUF) behind identical OpenAI-compatible REST endpoints (`/v1/completions`), containerized in reproducible Docker images and orchestrated via a single `Makefile` interface.

---

### Async Load Testing & Observability

> Captured P50/P95/P99 end-to-end latency, Time-To-First-Token (TTFT), and aggregate token throughput across concurrency levels 1–32, by building a streaming-aware async benchmarking harness (`asyncio` + `aiohttp`) that parses Server-Sent Events in real time to accurately measure TTFT independent of payload size, with a 2-minute per-request timeout guard to prevent indefinite hanging under server OOM conditions.

---

### Data Engineering & Results Provenance

> Produced a fully auditable, diffable results dataset across 6 model variants × 5 concurrency levels, by implementing an immutable telemetry architecture that writes every benchmark run as a timestamped, UUID-keyed JSON artifact to `results/raw/`, then aggregates them into a consolidated `results.csv` with automated Seaborn visualizations covering throughput, memory footprint, and quality-vs-speed tradeoff scatter plots.

---

### Cost-Aware Infrastructure Architecture

> Reduced cloud GPU compute costs for the quantization pipeline by isolating all CPU-compatible operations (GGUF conversion and quantization via llama.cpp + CMake) to local hardware while reserving expensive rented GPU instances (RunPod/Colab) exclusively for GPTQ/AWQ calibration and vLLM serving, a cost-aware split documented in the runbook as an explicit architectural decision.

---

### CI/CD & Engineering Rigor

> Maintained pipeline correctness across repository changes by implementing a GitHub Actions CI smoke test that validates the full download-quantize workflow end-to-end on a 1.1B-parameter TinyLlama model on every push to `main`, with CPU-optimized dependency installation and CMake-based llama.cpp compilation — proving the harness remains reproducible on a clean machine, not just the original dev box.

---

## 🎯 One-Liner Variants (for tight resume layouts)

- Benchmarked FP16, GPTQ, AWQ, and GGUF (Q4/Q5/Q8) quantization formats on a 7B LLM, measuring P50/P95/P99 latency, TTFT, token throughput, VRAM footprint, and Wikitext-2 perplexity across vLLM and llama.cpp runtimes via an async streaming load tester sweeping concurrency 1–32.

- Built a CI-tested, configuration-driven benchmark suite where every stage reads from `configs/*.yaml` and writes immutable JSON artifacts, making the entire pipeline reproducible from a single `make bench-all` command with zero manual state management.

- Designed a cost-aware quantization infrastructure that routes GGUF workloads to CPU-only hardware and GPTQ/AWQ calibration to rented GPU instances, lowering total benchmark compute cost while maintaining an apples-to-apples API comparison across runtimes.

---

## 🔑 Keyword Cloud (ATS Optimization)

`LLM Quantization` · `GPTQ` · `AWQ` · `GGUF` · `vLLM` · `llama.cpp` · `on-premise inference` · `model compression` · `4-bit quantization` · `perplexity evaluation` · `async load testing` · `aiohttp` · `asyncio` · `OpenAI API` · `Docker` · `NVIDIA CUDA` · `token throughput` · `TTFT` · `P95 latency` · `CMake` · `GitHub Actions CI/CD` · `Hugging Face Hub` · `AutoGPTQ` · `AutoAWQ` · `Wikitext-2` · `transformers` · `pandas` · `seaborn` · `infrastructure-as-code` · `MLOps` · `cost-aware infrastructure`
