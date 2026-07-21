.PHONY: help install bench-all download quantize-gptq quantize-awq quantize-gguf serve-vllm serve-llamacpp loadtest eval aggregate

help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  download       - Download base model"
	@echo "  quantize-gptq  - Quantize using GPTQ"
	@echo "  quantize-awq   - Quantize using AWQ"
	@echo "  quantize-gguf  - Quantize using GGUF"
	@echo "  bench-all      - Run the full benchmark pipeline"

install:
	pip install -e .

download:
	python scripts/01_download_base_model.py

quantize-gptq:
	python scripts/02_quantize_gptq.py

quantize-awq:
	python scripts/03_quantize_awq.py

quantize-gguf:
	bash scripts/04_quantize_gguf.sh

serve-vllm:
	bash scripts/05_serve_vllm.sh

serve-llamacpp:
	bash scripts/06_serve_llamacpp.sh

loadtest:
	python scripts/07_loadtest.py

eval:
	python scripts/08_eval_quality.py

aggregate:
	python scripts/09_aggregate_results.py

bench-all: download quantize-gptq quantize-awq quantize-gguf
	python scripts/run_benchmarks.py
