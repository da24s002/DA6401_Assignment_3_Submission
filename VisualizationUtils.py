import torch
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from SearchUtils import translate



def plot_detailed_confusion_matrix(true_chars, pred_chars, filename="confusion_matrix_detailed.png"):
    # Get unique characters (sorted for consistent ordering)
    unique_chars = sorted(list(set(true_chars + pred_chars)))
    
    # Create label mapping
    label_indices = {c: i for i, c in enumerate(unique_chars)}
    
    # Filter only chars present in labels and convert to indices
    filtered_true = []
    filtered_pred = []
    for t, p in zip(true_chars, pred_chars):
        if t in label_indices and p in label_indices:
            filtered_true.append(label_indices[t])
            filtered_pred.append(label_indices[p])
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred, labels=range(len(unique_chars)))
    
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=unique_chars, columns=unique_chars)
    cm_df.index.name = 'True'
    cm_df.columns.name = 'Predicted'
    
    # Create figure with larger size for detailed view
    plt.figure(figsize=(20, 18), dpi=300)
    
    # Plot heatmap with small font size for the values
    ax = sns.heatmap(cm_df, cmap='Blues', linewidths=0.5, 
                     annot=True, fmt='d', annot_kws={"size": 5})
    
    # Set title and labels
    plt.title('Character-level Confusion Matrix', fontsize=16)
    plt.ylabel('Ground Truth', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    
    # Adjust tick label size
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8, rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {filename}")
    
    return plt.gcf()  # Return the figure if needed for further use

def plot_attention_heatmaps(model, test_examples, input_vocab, output_vocab, device, num_examples=9, grid_size=(3, 3)):
    fig, axes = plt.subplots(*grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, example in enumerate(test_examples[:num_examples]):
        # Get prediction and attention weights
        pred_str, attn_weights = translate(
            model,
            example,
            input_vocab,
            output_vocab,
            device,
            decode_method="greedy",
            beam_width=5,
            length_penalty_alpha=1.0
        )
        
        # Convert input and output to character lists
        input_chars = list(example)
        output_chars = list(pred_str)
        
        ax = axes[i]
        
        if attn_weights is not None:
            attn_matrix = np.zeros((len(attn_weights) - 1, len(attn_weights[0][0]) - 1))
            for i in range(len(attn_matrix)):
                for j in range(len(attn_matrix[i])):
                    attn_matrix[i][j] = attn_weights[i][0][j + 1]

            
            # Plot heatmap
            im = ax.imshow(attn_matrix, cmap='Blues')
            
            # Set labels
            ax.set_xticks(range(len(input_chars)))
            ax.set_yticks(range(len(output_chars)))
            ax.set_xticklabels(input_chars)
            ax.set_yticklabels(output_chars)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "No attention weights available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
        
        # Add title
        ax.set_title(f"Input: {example}\nOutput: {pred_str}")
    
    plt.tight_layout()
    plt.savefig("attention_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return fig




# Load a few samples from your predictions file
def create_transliteration_grid(predictions_file, num_samples=10):
    # Read predictions
    samples = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Select samples (either first few or random)
        import random
        selected_lines = random.sample(lines, min(num_samples, len(lines)))
        
        for line in selected_lines:
            latin, pred, true = line.strip().split(',')
            # Check if prediction is correct
            is_correct = (pred == true)
            samples.append({
                'Latin Input': latin,
                'Model Output': pred,
                'Ground Truth': true,
                'Correct': is_correct
            })
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Hide axes
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Create table
    table = plt.table(
        cellText=df[['Latin Input', 'Model Output', 'Ground Truth']].values,
        colLabels=['Latin Input', 'Model Output', 'Ground Truth'],
        cellLoc='center',
        loc='center',
        cellColours=[[('lightgreen' if row['Correct'] else 'mistyrose') for _ in range(3)] for _, row in df.iterrows()]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Hindi Transliteration Results (LSTM Seq2Seq Model)', fontsize=16, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('transliteration_results.png', bbox_inches='tight', dpi=300)
    
    return fig, df





def visualize_attention_custom(input_word, output_word, attention_weights, save_path=None, figsize=(12, 8)):
    
    plt.rcParams['font.family'] = 'Nirmala UI'
    
    # Get dimensions
    output_len = len(output_word)
    input_len = len(input_word)
    
    # Create figure
    fig, axes = plt.subplots(output_len, 1, figsize=figsize)
    if output_len == 1:
        axes = [axes]
    
    # Title for the entire figure
    fig.suptitle(f"Input: {input_word} → Output: {output_word}", fontsize=16)
    
    # For each output character
    for i, output_char in enumerate(output_word):
        ax = axes[i]
        
        # Get attention weights for this output character
        weights = attention_weights[i]
        
        # Create a horizontal bar of input characters
        for j, input_char in enumerate(input_word):
            # Calculate color intensity based on attention weight
            color_intensity = weights[j]
            
            # Create rectangle with color based on attention weight
            rect = plt.Rectangle((j, 0), 1, 1, 
                               facecolor=plt.cm.Blues(color_intensity),
                               edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add the input character in the center of the rectangle
            ax.text(j + 0.5, 0.5, input_char, 
                   ha='center', va='center', 
                   fontsize=14, 
                   color='black' if color_intensity < 0.5 else 'white')
            
        
        # Add the output character on the left
        ax.text(-0.5, 0.5, output_char, ha='center', va='center', fontsize=16)
        
        # Set axis limits and remove ticks
        ax.set_xlim(-1, input_len)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig




def visualize_word_attention(model, input_word, input_vocab, output_vocab, device, save_path=None):
    
    # Get prediction and attention weights
    model.eval()
    with torch.no_grad():
        pred_str, attention_weights = translate(
            model,
            input_word,
            input_vocab,
            output_vocab,
            device,
            decode_method="greedy"
        )
    
    if attention_weights is None:
        print("No attention weights available for this model.")
        return None
    
    # Process attention weights
    attention_matrix = []
    for weights in attention_weights:
        # Skip the first token (SOS)
        attention_matrix.append(weights[0][1:])
    attention_matrix = np.array(attention_matrix)
    
    return visualize_attention_custom(
        input_word, 
        pred_str, 
        attention_matrix, 
        save_path=save_path
    )