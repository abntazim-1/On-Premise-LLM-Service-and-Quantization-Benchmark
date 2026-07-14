import os
import json
import torch
import argparse
import subprocess
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def calculate_perplexity_hf(model, tokenizer, dataset, max_length=512, limit=50):
    # Join a subset of the dataset to create a long context string
    text = "\n\n".join(dataset['text'][:limit])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    stride = max_length // 2
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating HF Perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Ignore context for loss

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def calculate_perplexity_gguf(model_path, dataset, limit=50):
    """Uses llama.cpp's native perplexity tool."""
    if not os.path.exists("llama.cpp/llama-perplexity"):
        print("llama-perplexity binary not found. Building it...")
        subprocess.run(["make", "-C", "llama.cpp", "llama-perplexity"], check=True)
        
    text = "\n\n".join(dataset['text'][:limit])
    temp_file = "eval/wiki_temp.txt"
    
    os.makedirs("eval", exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"Shelling out to llama-perplexity for {model_path}...")
    try:
        cmd = ["./llama.cpp/llama-perplexity", "-m", model_path, "-f", temp_file]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse PPL from output. Format is usually "Final estimate: PPL = X.XX"
        output_lines = res.stdout.split('\n') + res.stderr.split('\n')
        for line in output_lines:
            if "PPL =" in line:
                return float(line.split("PPL =")[-1].strip())
    except Exception as e:
        print(f"GGUF Perplexity evaluation failed: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality (Perplexity)")
    parser.add_argument("--variant", type=str, required=True, help="Variant name: base, gptq, awq, or q4_k_m, etc.")
    args = parser.parse_args()

    print("Loading Wikitext-2 dataset for evaluation...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    ppl = 0.0
    if args.variant in ["fp16", "base", "gptq", "awq"]:
        # HF based models
        model_path = "models/base_model" if args.variant in ["fp16", "base"] else f"models/{args.variant}_4bit"
        
        print(f"Loading Hugging Face model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        ppl = calculate_perplexity_hf(model, tokenizer, dataset, limit=100)
        
    else:
        # GGUF models
        model_path = f"models/gguf/model-{args.variant.upper()}.gguf"
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found.")
            return
            
        ppl = calculate_perplexity_gguf(model_path, dataset, limit=100)
        
    print(f"\n--- Quality Result ---")
    print(f"[{args.variant}] Wikitext-2 Perplexity: {ppl:.4f}")
    
    # Save the quality score to raw results
    os.makedirs("results/raw", exist_ok=True)
    out_file = f"results/raw/quality_{args.variant}.json"
    with open(out_file, "w") as f:
        json.dump({
            "variant": args.variant, 
            "perplexity_wikitext2": ppl
        }, f, indent=2)
    print(f"Saved metric to {out_file}")
    
    # Phase 5: Generate Human Grading Rubric using task prompts
    prompts_file = "eval/task_prompts.jsonl"
    if os.path.exists(prompts_file):
        print("\nGenerating task responses for human grading rubric...")
        prompts = []
        with open(prompts_file, "r") as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line)["text"])
        
        rubric_out = f"results/raw/rubric_eval_{args.variant}.md"
        with open(rubric_out, "w", encoding="utf-8") as f:
            f.write(f"# Human Grading Rubric for {args.variant}\n\n")
            f.write("Score each prompt on a scale of 1-5 for accuracy, coherence, and instruction adherence.\n\n")
            
            for i, prompt in enumerate(prompts):
                print(f"Generating prompt {i+1}/{len(prompts)}...")
                f.write(f"## Prompt {i+1}\n**Q**: {prompt}\n\n")
                
                answer = ""
                if args.variant in ["fp16", "base", "gptq", "awq"]:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
                    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                else:
                    cmd = ["./llama.cpp/llama-cli", "-m", model_path, "-p", prompt, "-n", "150", "--log-disable"]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        answer = res.stdout.replace(prompt, "").strip()
                    except Exception as e:
                        answer = f"[Error generating response: {e}]"
                
                f.write(f"**A**: {answer}\n\n")
                f.write(f"**Score (1-5)**: ___\n\n---\n")
                
        print(f"Saved human grading rubric to {rubric_out}")

if __name__ == "__main__":
    main()
