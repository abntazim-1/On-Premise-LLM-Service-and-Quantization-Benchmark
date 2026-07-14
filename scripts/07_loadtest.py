import os
import time
import json
import uuid
import yaml
import asyncio
import aiohttp
import argparse
import random
import statistics
import psutil
from datetime import datetime

def load_config():
    with open("configs/loadtest.yaml", "r") as f:
        return yaml.safe_load(f)

def load_prompts(file_path):
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["text"])
    return prompts

def get_checkpoint_size_gb(variant):
    path = ""
    if variant in ["fp16", "base"]:
        path = "models/base_model"
    elif variant in ["gptq", "awq"]:
        path = f"models/{variant}_4bit"
    else:
        path = f"models/gguf/model-{variant.upper()}.gguf"
        
    if not os.path.exists(path):
        return 0.0
        
    total_size = 0
    if os.path.isfile(path):
        total_size = os.path.getsize(path)
    else:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
                    
    return total_size / (1024**3)

async def get_gpu_memory():
    """Simple nvidia-smi wrapper to get GPU memory used."""
    try:
        proc = await asyncio.create_subprocess_shell(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().strip().split('\n')
        return [int(x) for x in lines if x.isdigit()]
    except Exception:
        return []

async def worker(worker_id, session, api_url, model_name, prompts, run_state, stop_event, is_warmup=False):
    while not stop_event.is_set():
        prompt = random.choice(prompts)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": True
        }
        
        start_time = time.time()
        ttft = None
        total_tokens = 0
        
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    if not is_warmup: run_state['errors'] += 1
                    continue
                
                # Process streaming response for TTFT
                async for line in response.content:
                    if stop_event.is_set():
                        break
                        
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if chunk.get("choices") and chunk["choices"][0].get("text"):
                                    total_tokens += 1
                                    if ttft is None:
                                        ttft = time.time() - start_time
                            except json.JSONDecodeError:
                                pass
                                
            if not is_warmup:
                latency = time.time() - start_time
                run_state['latencies'].append(latency)
                if ttft is not None:
                    run_state['ttfts'].append(ttft)
                run_state['tokens'] += total_tokens
                run_state['requests'] += 1
            
        except Exception as e:
            if not is_warmup: run_state['errors'] += 1
            await asyncio.sleep(0.1) # backoff

async def monitor_resources(run_state, stop_event):
    while not stop_event.is_set():
        # RAM Usage
        run_state['peak_ram_mb'] = max(
            run_state.get('peak_ram_mb', 0), 
            psutil.virtual_memory().used / (1024 * 1024)
        )
        # VRAM Usage
        gpus = await get_gpu_memory()
        if gpus:
            run_state['peak_vram_mb'] = max(
                run_state.get('peak_vram_mb', 0),
                sum(gpus)
            )
        await asyncio.sleep(0.5)

async def run_loadtest(concurrency, duration, prompts, config, variant, runtime, api_url):
    print(f"\n--- Starting test: {variant} on {runtime} at concurrency {concurrency} ---")
    model_name = config.get("model_name", "default_model")
    
    async with aiohttp.ClientSession() as session:
        # 1. Warm-up Phase
        print("Warming up server (5 seconds) to compile CUDA graphs and init KV cache...")
        warmup_stop = asyncio.Event()
        warmup_workers = [
            asyncio.create_task(worker(i, session, api_url, model_name, prompts, {}, warmup_stop, is_warmup=True))
            for i in range(min(4, concurrency))
        ]
        await asyncio.sleep(5)
        warmup_stop.set()
        await asyncio.gather(*warmup_workers, return_exceptions=True)
        
        # 2. Main Benchmark Phase
        print("Starting active benchmarking phase...")
        run_state = {
            'requests': 0, 'errors': 0, 'tokens': 0, 'latencies': [], 'ttfts': [],
            'peak_ram_mb': 0, 'peak_vram_mb': 0
        }
        
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_resources(run_state, stop_event))
        
        workers = [
            asyncio.create_task(worker(i, session, api_url, model_name, prompts, run_state, stop_event, is_warmup=False))
            for i in range(concurrency)
        ]
        
        await asyncio.sleep(duration)
        stop_event.set() 
        
        await asyncio.gather(*workers, return_exceptions=True)
        await monitor_task
        
    # Calculate summary metrics
    lats = run_state['latencies']
    ttfts = run_state['ttfts']
    
    metrics = {
        "variant": variant,
        "runtime": runtime,
        "concurrency": concurrency,
        "duration_s": duration,
        "total_requests": run_state['requests'],
        "total_errors": run_state['errors'],
        "total_tokens": run_state['tokens'],
        "throughput_tok_s": run_state['tokens'] / duration if duration > 0 else 0,
        "latency_ms_p50": statistics.median(lats) * 1000 if lats else 0,
        "latency_ms_p95": statistics.quantiles(lats, n=20)[18] * 1000 if len(lats) > 1 else 0,
        "latency_ms_p99": statistics.quantiles(lats, n=100)[98] * 1000 if len(lats) > 1 else 0,
        "ttft_ms_p50": statistics.median(ttfts) * 1000 if ttfts else 0,
        "ttft_ms_p95": statistics.quantiles(ttfts, n=20)[18] * 1000 if len(ttfts) > 1 else 0,
        "peak_ram_mb": run_state['peak_ram_mb'],
        "peak_vram_mb": run_state['peak_vram_mb'],
        "checkpoint_size_gb": get_checkpoint_size_gb(variant),
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": str(uuid.uuid4())
    }
    
    print(f"Results for concurrency {concurrency}:")
    print(f"  Requests: {metrics['total_requests']} (Errors: {metrics['total_errors']})")
    print(f"  Throughput: {metrics['throughput_tok_s']:.2f} tokens/sec")
    print(f"  P50 Latency: {metrics['latency_ms_p50']:.2f} ms")
    print(f"  P50 TTFT: {metrics['ttft_ms_p50']:.2f} ms")
    print(f"  Peak VRAM: {metrics['peak_vram_mb']:.2f} MB (Note: this is whole-system VRAM)")
    print(f"  Model Size: {metrics['checkpoint_size_gb']:.2f} GB")
    
    out_dir = "results/raw"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{variant}_{runtime}_c{concurrency}_{metrics['run_id']}.json")
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved raw metrics to {out_file}")

async def main():
    parser = argparse.ArgumentParser(description="LLM Load Tester")
    parser.add_argument("--variant", type=str, required=True, help="Model variant (e.g. fp16, gptq, q4_k_m)")
    parser.add_argument("--runtime", type=str, required=True, help="Runtime used (e.g. vllm, llamacpp)")
    parser.add_argument("--api-url", type=str, help="Override API URL", default=None)
    args = parser.parse_args()

    config = load_config()
    prompts_file = config.get("prompts_file", "eval/task_prompts.jsonl")
    
    if not os.path.exists(prompts_file):
        print(f"Error: Prompt file {prompts_file} not found.")
        return
        
    prompts = load_prompts(prompts_file)
    duration = config.get("duration_seconds", 60)
    concurrencies = config.get("concurrency_levels", [1, 4, 8, 16, 32])
    api_url = args.api_url or config.get("api_url", "http://localhost:8000/v1/completions")
    
    print(f"Starting benchmark for {args.variant} on {args.runtime}")
    print(f"API URL: {api_url}")
    print(f"Sweeping concurrencies: {concurrencies} over {duration} seconds each")
    
    for c in concurrencies:
        await run_loadtest(c, duration, prompts, config, args.variant, args.runtime, api_url)

if __name__ == "__main__":
    asyncio.run(main())
