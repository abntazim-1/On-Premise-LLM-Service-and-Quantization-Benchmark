import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

def load_config():
    with open("configs/quant_awq.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_dir = "models/base_model"
    quant_dir = "models/awq_4bit"
    
    if not os.path.exists(model_dir):
        print(f"Error: Base model not found at {model_dir}. Please run download script first.")
        return

    os.makedirs(quant_dir, exist_ok=True)
    
    print(f"Loading model and tokenizer from {model_dir} for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    quant_config = {
        "zero_point": config.get("zero_point", True),
        "q_group_size": config.get("q_group_size", 128),
        "w_bit": config.get("w_bit", 4),
        "version": config.get("version", "GEMM")
    }
    
    print(f"Starting AWQ quantization...")
    print(f"Configuration: {quant_config}")
    print("This will take some time and requires a GPU.")
    
    # AutoAWQ uses MIT Han Lab's calib data implicitly if not provided, or pulls wikitext.
    model.quantize(tokenizer, quant_config=quant_config)
    
    print(f"Saving AWQ quantized model to {quant_dir}...")
    model.save_quantized(quant_dir)
    tokenizer.save_pretrained(quant_dir)
    
    print("AWQ Quantization complete!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. AWQ quantization requires a GPU. Exiting.")
        exit(1)
    main()
