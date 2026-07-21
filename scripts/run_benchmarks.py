import os
import sys
import time
import subprocess
import torch

# Ensure we are in the project root directory
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def clean_up_container():
    print("Cleaning up any existing benchmark containers...")
    subprocess.run(["docker", "stop", "llm-benchmark-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "llm-benchmark-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for OS to release ports and CUDA to clear
    time.sleep(3)

def run_benchmark(variant, runtime, serve_script, serve_arg, port="8000"):
    print(f"\n==================================================")
    print(f"🚀 Running Benchmark: {variant} via {runtime}")
    print(f"==================================================")
    
    clean_up_container()
    
    # 1. Start Server in Background
    print(f"Starting server for {variant}...")
    serve_cmd = ["bash", serve_script, serve_arg, port]
    res = subprocess.run(serve_cmd)
    if res.returncode != 0:
        print(f"❌ Failed to start server for {variant}.")
        return False
        
    # 2. Run Loadtest
    print(f"Running loadtest for {variant}...")
    loadtest_cmd = [
        "python", "scripts/07_loadtest.py",
        "--variant", variant,
        "--runtime", runtime,
        "--api-url", f"http://localhost:{port}/v1/completions"
    ]
    res = subprocess.run(loadtest_cmd)
    if res.returncode != 0:
        print(f"❌ Loadtest failed for {variant}.")
        # Continue to cleanup anyway
        
    # 3. Stop Serving Container to Release GPU VRAM and Memory
    print("Stopping and removing serving container to free memory...")
    clean_up_container()
    
    # 4. Run Quality Evaluation
    print(f"Evaluating quality metrics (Perplexity & Rubric) for {variant}...")
    eval_cmd = [
        "python", "scripts/08_eval_quality.py",
        "--variant", variant
    ]
    res = subprocess.run(eval_cmd)
    if res.returncode != 0:
        print(f"❌ Quality evaluation failed for {variant}.")
        
    print(f"✅ Finished benchmark for {variant}!\n")
    return True

def main():
    has_gpu = torch.cuda.is_available()
    
    gpu_benchmarks = [
        ("base", "vllm", "scripts/05_serve_vllm.sh", "base"),
        ("gptq", "vllm", "scripts/05_serve_vllm.sh", "gptq"),
        ("awq", "vllm", "scripts/05_serve_vllm.sh", "awq"),
    ]
    
    cpu_benchmarks = [
        ("q4_k_m", "llamacpp", "scripts/06_serve_llamacpp.sh", "q4_k_m"),
        ("q5_k_m", "llamacpp", "scripts/06_serve_llamacpp.sh", "q5_k_m"),
        ("q8_0", "llamacpp", "scripts/06_serve_llamacpp.sh", "q8_0"),
    ]
    
    print("Starting LLM Serving & Quantization Benchmarking Suite Orchestrator")
    print(f"GPU Available: {has_gpu}")
    
    # Run GPU benchmarks if hardware allows
    if has_gpu:
        print("\n--- Running GPU vLLM Benchmarks ---")
        for variant, runtime, script, arg in gpu_benchmarks:
            run_benchmark(variant, runtime, script, arg)
    else:
        print("\n⚠️ No GPU detected. Skipping GPU-bound vLLM (Base, GPTQ, AWQ) benchmarks.")
        
    # Run CPU GGUF benchmarks
    print("\n--- Running CPU llama.cpp GGUF Benchmarks ---")
    for variant, runtime, script, arg in cpu_benchmarks:
        run_benchmark(variant, runtime, script, arg)
        
    # Finally, run aggregation to generate the consolidated report and charts
    print("\n==================================================")
    print("📊 Aggregating all benchmark results...")
    print("==================================================")
    subprocess.run(["python", "scripts/09_aggregate_results.py"])
    
    print("\n🎉 Benchmarking suite complete! Results are ready in results/aggregated/results.csv and charts are in results/charts/")

if __name__ == "__main__":
    main()
