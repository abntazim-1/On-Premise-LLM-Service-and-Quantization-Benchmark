import os
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_config():
    with open("configs/model.yaml", "r") as f:
        return yaml.safe_load(f)

def download_model(model_id, revision, local_dir):
    print(f"Downloading model {model_id} (revision: {revision}) to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.safetensors.index.json"], # Exclude unnecessary files
        local_dir_use_symlinks=False
    )
    print("Download complete.")

def test_inference(local_dir):
    print("Loading model for a baseline inference test...")
    # Load with float16 to save memory, as it's the fp16 baseline
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "The future of artificial intelligence is"
    
    print(f"Running inference on prompt: '{prompt}'")
    result = pipe(prompt, max_new_tokens=20, do_sample=True, top_k=50, top_p=0.95)
    
    print("\n--- Baseline Inference Result ---")
    print(result[0]['generated_text'])
    print("---------------------------------")
    
    # Capture rough VRAM usage if on GPU
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU VRAM Allocated: {allocated:.2f} GB")
        print(f"GPU VRAM Reserved:  {reserved:.2f} GB")

if __name__ == "__main__":
    config = load_config()
    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    local_dir = "models/base_model"
    
    os.makedirs(local_dir, exist_ok=True)
    
    download_model(model_id, revision, local_dir)
    
    # Optionally run inference test if torch is available
    if torch.cuda.is_available():
        test_inference(local_dir)
    else:
        print("CUDA not available. Skipping baseline inference test to avoid long CPU delays.")
