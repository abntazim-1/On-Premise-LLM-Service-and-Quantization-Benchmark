import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

def load_config():
    with open("configs/quant_gptq.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_dir = "models/base_model"
    quant_dir = "models/gptq_4bit"
    
    if not os.path.exists(model_dir):
        print(f"Error: Base model not found at {model_dir}. Please run download script first.")
        return

    os.makedirs(quant_dir, exist_ok=True)
    
    print("Loading tokenizer and calibration dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load calibration dataset (e.g., wikitext2)
    dataset_name = config.get("dataset", "wikitext")
    dataset_config = "wikitext-2-raw-v1" if dataset_name == "wikitext2" else None
    
    if dataset_config:
        dataset = load_dataset("wikitext", dataset_config, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Prepare calibration samples
    samples = []
    num_samples = config.get("samples", 128)
    for data in dataset:
        if len(data["text"]) > 100:  # Skip empty or very short lines
            # Tokenize and create expected dict structure
            tokenized = tokenizer(data["text"], return_tensors="pt")
            samples.append({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]})
            if len(samples) >= num_samples:
                break

    quantize_config = BaseQuantizeConfig(
        bits=config.get("bits", 4),
        group_size=config.get("group_size", 128),
        desc_act=config.get("desc_act", False),
        sym=config.get("sym", True)
    )
    
    print(f"Loading base model from {model_dir} for GPTQ quantization...")
    # AutoGPTQ requires the model to be loaded via its class
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config)
    
    print(f"Starting GPTQ quantization ({num_samples} calibration samples)...")
    print("This will take some time and requires a GPU.")
    
    model.quantize(samples)
    
    print(f"Saving GPTQ quantized model to {quant_dir}...")
    model.save_quantized(quant_dir)
    tokenizer.save_pretrained(quant_dir)
    
    # Save a copy of the config inside the dir for reference
    with open(os.path.join(quant_dir, "quant_config.json"), "w") as f:
        f.write(quantize_config.to_json_string())

    print("GPTQ Quantization complete!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. GPTQ quantization requires a GPU. Exiting.")
        exit(1)
    main()
