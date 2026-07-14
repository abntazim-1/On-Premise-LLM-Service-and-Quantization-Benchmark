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

async def worker(worker_id, session, config, prompts, run_state, stop_event):
    api_url = config.get("api_url", "http://localhost:8000/v1/completions")
    
    while not stop_event.is_set():
        prompt = random.choice(prompts)
        payload = {
            "model": config.get("model_name", "default_model"),
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
                    run_state['errors'] += 1
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
                                
            end_time = time.time()
            latency = end_time - start_time
            
            run_state['latencies'].append(latency)
            if ttft is not None:
                run_state['ttfts'].append(ttft)
            run_state['tokens'] += total_tokens
            run_state['requests'] += 1
            
        except Exception as e:
            run_state['errors'] += 1
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

async def run_loadtest(concurrency, duration, prompts, config, variant, runtime):
    print(f"\n--- Starting test: {variant} on {runtime} at concurrency {concurrency} ---")
    
    run_state = {
        'requests': 0,
        'errors': 0,
        'tokens': 0,
        'latencies': [],
        'ttfts': [],
        'peak_ram_mb': 0,
        'peak_vram_mb': 0
    }
    
    stop_event = asyncio.Event()
    
    async with aiohttp.ClientSession() as session:
        # Start resource monitor
        monitor_task = asyncio.create_task(monitor_resources(run_state, stop_event))
        
        # Start async workers
        workers = [
            asyncio.create_task(worker(i, session, config, prompts, run_state, stop_event))
            for i in range(concurrency)
        ]
        
        # Wait for the specified test duration
        await asyncio.sleep(duration)
        stop_event.set() # Signal workers to stop
        
        # Wait for all workers to gracefully finish their current request
        await asyncio.gather(*workers)
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
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": str(uuid.uuid4())
    }
    
    print(f"Results for concurrency {concurrency}:")
    print(f"  Requests: {metrics['total_requests']} (Errors: {metrics['total_errors']})")
    print(f"  Throughput: {metrics['throughput_tok_s']:.2f} tokens/sec")
    print(f"  P50 Latency: {metrics['latency_ms_p50']:.2f} ms")
    print(f"  P50 TTFT: {metrics['ttft_ms_p50']:.2f} ms")
    print(f"  Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
    
    # Save raw JSON results directly to disk as an immutable artifact
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
    args = parser.parse_args()

    config = load_config()
    prompts_file = config.get("prompts_file", "eval/task_prompts.jsonl")
    
    if not os.path.exists(prompts_file):
        print(f"Error: Prompt file {prompts_file} not found.")
        return
        
    prompts = load_prompts(prompts_file)
    duration = config.get("duration_seconds", 60)
    concurrencies = config.get("concurrency_levels", [1, 4, 8, 16, 32])
    
    print(f"Starting benchmark for {args.variant} on {args.runtime}")
    print(f"API URL: {config.get('api_url')}")
    print(f"Sweeping concurrencies: {concurrencies} over {duration} seconds each")
    
    for c in concurrencies:
        await run_loadtest(c, duration, prompts, config, args.variant, args.runtime)

if __name__ == "__main__":
    asyncio.run(main())
