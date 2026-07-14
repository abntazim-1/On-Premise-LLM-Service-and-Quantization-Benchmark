import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    raw_dir = "results/raw"
    agg_dir = "results/aggregated"
    charts_dir = "results/charts"
    
    os.makedirs(agg_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    data = []
    quality_data = {}
    
    if not os.path.exists(raw_dir):
        print(f"No results found in {raw_dir}")
        return

    print("Parsing raw JSON results...")
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".json"): continue
        
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r") as file:
            content = json.load(file)
            
            # Separate quality metrics from load test metrics
            if filename.startswith("quality_"):
                quality_data[content["variant"]] = content.get("perplexity_wikitext2", 0)
            else:
                data.append(content)
                
    if not data:
        print("No load test metrics found to aggregate.")
        return
        
    df = pd.DataFrame(data)
    
    # Map quality data into the main dataframe
    df["perplexity_wikitext2"] = df["variant"].map(lambda v: quality_data.get(v, None))
    
    # Sort for predictable CSV structure
    df = df.sort_values(["variant", "concurrency"])
    
    # 1. Save Aggregated CSV
    csv_path = os.path.join(agg_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to {csv_path}")
    
    # 2. Generate Visualizations
    print("Generating charts...")
    sns.set_theme(style="whitegrid")
    
    # Chart A: Throughput vs Concurrency
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="concurrency", 
        y="throughput_tok_s", 
        hue="variant", 
        style="runtime", 
        markers=True,
        dashes=False
    )
    plt.title("Throughput vs Concurrency")
    plt.ylabel("Throughput (Tokens / Second)")
    plt.xlabel("Concurrency (Simultaneous Requests)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "throughput.png"), dpi=300)
    plt.close()
    
    # Chart B: Memory Footprint
    # Use concurrency=1 as memory footprint is roughly stable across test runs for our setup
    plt.figure(figsize=(10, 6))
    mem_df = df[df["concurrency"] == 1]
    if not mem_df.empty:
        sns.barplot(data=mem_df, x="variant", y="peak_vram_mb", hue="runtime")
        plt.title("Peak VRAM Footprint by Variant (Concurrency = 1)")
        plt.ylabel("Peak VRAM (MB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "memory.png"), dpi=300)
    plt.close()
    
    # Chart C: Quality vs Speed Tradeoff
    plt.figure(figsize=(10, 6))
    # Pick a standardized concurrency level (e.g., 16) for the tradeoff comparison
    tradeoff_df = df[df["concurrency"] == 16].copy()
    
    if not tradeoff_df.empty and tradeoff_df["perplexity_wikitext2"].notnull().any():
        sns.scatterplot(
            data=tradeoff_df, 
            x="throughput_tok_s", 
            y="perplexity_wikitext2",
            hue="variant",
            s=200,
            alpha=0.8
        )
        
        # Annotate scatter points with the variant names
        for i in range(tradeoff_df.shape[0]):
            plt.text(
                tradeoff_df["throughput_tok_s"].iloc[i] * 1.02, 
                tradeoff_df["perplexity_wikitext2"].iloc[i], 
                tradeoff_df["variant"].iloc[i],
                fontsize=9
            )
            
        plt.title("Quality vs Speed Tradeoff (Concurrency = 16)")
        plt.xlabel("Throughput (Tokens / Sec) ➔ Higher is better")
        plt.ylabel("Perplexity (Wikitext-2) ➔ Lower is better")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "quality_tradeoff.png"), dpi=300)
    plt.close()
    
    print(f"Charts saved to {charts_dir}")

if __name__ == "__main__":
    main()
