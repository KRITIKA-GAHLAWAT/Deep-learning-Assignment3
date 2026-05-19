import torch
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from model import Transformer

def main():
    # Initialize W&B run for Section 2.3
    wandb.init(project="da6401-a3", name="attention_visualization_2.3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model using your existing checkpoint (no training needed!)
    print("Loading model from checkpoint...")
    model = Transformer(checkpoint_path="checkpoint.pt").to(device)
    model.eval()
    
    # A sample German sentence (Encoder processes the source language)
    # Translation: "A small dog is running across the grass."
    src_sentence = "Ein kleiner Hund rennt über das Gras."
    
    # Run inference to populate the attention weights in the model
    print(f"Running inference for: '{src_sentence}'")
    translation = model.infer(src_sentence)
    print(f"Translation Output: '{translation}'")
    
    # Get the tokens for the X and Y axes of the heatmap
    tokens = model.dataset_helper.tokenize_de(src_sentence)
    display_tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Extract attention weights from the LAST encoder layer
    # `last_attn_weights` shape: [batch_size=1, num_heads=8, seq_len, seq_len]
    attn_weights = model.encoder.layers[-1].self_attn.last_attn_weights[0].cpu().numpy()
    
    num_heads = attn_weights.shape[0]
    
    # Plot a 2x4 grid of heatmaps (for 8 heads)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(num_heads):
        ax = axes[i]
        sns.heatmap(attn_weights[i], ax=ax, cmap="viridis", 
                    xticklabels=display_tokens, yticklabels=display_tokens,
                    cbar=False)
        ax.set_title(f"Head {i+1}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
    plt.tight_layout()
    
    # Save the figure locally and log it to Weights & Biases
    plt.savefig("attention_heads.png")
    wandb.log({"encoder_attention_heatmap": wandb.Image("attention_heads.png")})
    print("Successfully logged attention heatmap to W&B!")

if __name__ == "__main__":
    main()
