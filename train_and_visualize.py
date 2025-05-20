import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb

from tqdm import tqdm

from SearchUtils import translate

from Seq2SeqRNN import Seq2SeqRNN
from Seq2SeqAttention import Seq2SeqAttention
from VocabUtils import DakshinaLexiconDataset

import os


from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import torch
import matplotlib.font_manager as fm

from VisualizationUtils import plot_detailed_confusion_matrix, plot_attention_heatmaps,  create_transliteration_grid, visualize_word_attention


#supports Devanagari
plt.rcParams['font.family'] = 'Nirmala UI'  # Or another font from your list

def collate_fn(batch):
    srcs, targets = zip(*batch)
    src_lens = [len(s) for s in srcs]
    target_lens = [len(t) for t in targets]
    max_src = max(src_lens)
    max_target = max(target_lens)
    pad_srcs = [s + [0]*(max_src-len(s)) for s in srcs]
    pad_targets = [t + [0]*(max_target-len(t)) for t in targets]
    return (torch.tensor(pad_srcs, dtype=torch.long), torch.tensor(pad_targets, dtype=torch.long), src_lens, target_lens)


def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    
    # Use tqdm for progress bar
    for src, target, src_lens, target_lens in tqdm(dataloader, desc="Training"):
        src, target = src.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(src, target)
        # Shift targets to ignore <sos>
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_combined(model, dataloader, criterion, device, input_vocab, output_vocab, decode_method='greedy', beam_width=5, length_penalty_alpha=0.75):
    model.eval()
    total_loss = 0
    
    # Character accuracy metrics
    total_correct_chars = 0
    total_chars = 0
    
    # Sequence accuracy metrics
    total_correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for src, target, src_lens, target_lens in tqdm(dataloader, desc="Evaluating"):
            src, target = src.to(device), target.to(device)
            batch_size = src.shape[0]
            
            # Forward pass with no teacher forcing for loss calculation
            output = model(src, target, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
            
            # For accuracy, evaluate each example separately with the specified decoding method
            for i in range(batch_size):
                src_i = src[i].unsqueeze(0)
                target_i = target[i]
                
                # Convert source tensor back to a sentence
                src_sentence = input_vocab.decode([idx.item() for idx in src_i[0] if idx.item() != input_vocab.pad_idx])
                
                # Get predictions using the specified decoding method
                pred_sequence, _ = translate(
                    model, 
                    src_sentence, 
                    input_vocab, 
                    output_vocab, 
                    device, 
                    decode_method=decode_method,
                    beam_width=beam_width,
                    length_penalty_alpha=length_penalty_alpha
                )
                
                # Convert prediction to indices for comparison
                pred_indices = output_vocab.encode(pred_sequence)
                
                # Get target sequence (excluding padding)
                target_seq = [t.item() for t in target_i if t.item() != output_vocab.pad_idx]
                
                # Skip the SOS token for comparison
                pred_seq = pred_indices[1:] if len(pred_indices) > 0 else []
                target_seq = target_seq[1:] if len(target_seq) > 0 else []
                
                # Calculate character accuracy
                min_len = min(len(pred_seq), len(target_seq))
                correct_chars = sum(1 for j in range(min_len) if pred_seq[j] == target_seq[j])
                total_correct_chars += correct_chars
                total_chars += len(target_seq)
                
                # Check sequence accuracy (all characters match)
                if pred_seq == target_seq:
                    total_correct_sequences += 1
                total_sequences += 1
    
    # Calculate both accuracy metrics
    char_acc = total_correct_chars / total_chars if total_chars > 0 else 0
    seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0
    
    return total_loss / len(dataloader), char_acc, seq_acc


# def evaluate_combined(model, dataloader, criterion, device, input_vocab, output_vocab, decode_method='greedy', beam_width=5, alpha=1.0, max_len=50):
#     model.eval()
#     total_loss = 0
    
#     # Character accuracy metrics
#     total_correct_chars = 0
#     total_chars = 0
    
#     # Sequence accuracy metrics
#     total_correct_sequences = 0
#     total_sequences = 0
    
#     with torch.no_grad():
#         for src, target, src_lens, target_lens in tqdm(dataloader, desc="Evaluating"):
#             src, target = src.to(device), target.to(device)
#             batch_size = src.shape[0]
            
#             # Calculate loss using teacher forcing (standard approach)
#             output = model(src, target, teacher_forcing_ratio=0.0)
#             output_dim = output.shape[-1]
#             output_flat = output[:, 1:].reshape(-1, output_dim)
#             target_flat = target[:, 1:].reshape(-1)
#             loss = criterion(output_flat, target_flat)
#             total_loss += loss.item()
            
#             # For accuracy, use the specified decoding method
#             for i in range(batch_size):
#                 # Get the input sequence
#                 input_seq = src[i].cpu().numpy()
#                 # Remove padding
#                 input_seq = [idx for idx in input_seq if idx != output_vocab.pad_idx]
#                 # Convert to string
#                 input_str = ''.join([output_vocab.itos[idx] if idx < len(output_vocab.itos) else '' for idx in input_seq])
#                 input_str = input_str.replace('<sos>', '').replace('<eos>', '').replace('<pad>', '')
                
#                 # Get the target sequence
#                 target_seq = target[i].cpu().numpy()
#                 # Remove padding
#                 target_seq = [idx for idx in target_seq if idx != output_vocab.pad_idx]
#                 # Skip <sos> token
#                 if len(target_seq) > 0 and target_seq[0] == output_vocab.sos_idx:
#                     target_seq = target_seq[1:]
#                 # Convert to string
#                 target_str = output_vocab.decode(target_seq)
                
#                 # Generate prediction using the specified decoding method
#                 if decode_method == 'beam':
#                     pred_str = beam_search_decode(model, input_str, input_vocab, output_vocab, device, beam_width, max_len, alpha)
#                 else:  # greedy
#                     pred_str = greedy_decode(model, input_str, input_vocab, output_vocab, device, max_len)
                
#                 # Convert prediction to sequence of indices
#                 pred_seq = output_vocab.encode(pred_str)
#                 # Remove <sos> and <eos>
#                 if len(pred_seq) > 0 and pred_seq[0] == output_vocab.sos_idx:
#                     pred_seq = pred_seq[1:]
#                 if len(pred_seq) > 0 and pred_seq[-1] == output_vocab.eos_idx:
#                     pred_seq = pred_seq[:-1]
                
#                 # Calculate character accuracy
#                 min_len = min(len(target_seq), len(pred_seq))
#                 for j in range(min_len):
#                     if target_seq[j] == pred_seq[j]:
#                         total_correct_chars += 1
#                 total_chars += len(target_seq)
                
#                 # Calculate sequence accuracy
#                 sequence_correct = (len(target_seq) == len(pred_seq)) and all(t == p for t, p in zip(target_seq, pred_seq))
#                 if sequence_correct:
#                     total_correct_sequences += 1
#                 total_sequences += 1
    
#     # Calculate both accuracy metrics
#     char_acc = total_correct_chars / total_chars if total_chars > 0 else 0
#     seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0
    
#     return total_loss / len(dataloader), char_acc, seq_acc

# def translate(model, sentence, input_vocab, output_vocab, device, max_len=50):
#     model.eval()
    
#     # Convert sentence to indices
#     tokens = input_vocab.encode(sentence)
#     src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
#     # Get encoder outputs and hidden state
#     with torch.no_grad():
#         embedded = model.embedding(src_tensor)
#         encoder_outputs, encoder_hidden = model.encoder(embedded)
        
#         if model.encoder_rnn_type == 'LSTM':
#             hidden, cell = encoder_hidden
#         else:
#             hidden = encoder_hidden
#             cell = None
            
#         # Start with <sos> token
#         input = torch.tensor([output_vocab.sos_idx], device=device)
        
#         hidden_dec = hidden
#         cell_dec = cell
        
#         outputs = [output_vocab.sos_idx]
#         attentions = []
        
#         for i in range(max_len):
#             # Get embedding
#             # input_emb = model.decoder_embedding(input).unsqueeze(0).unsqueeze(0)
#             input_emb = model.decoder_embedding(input).unsqueeze(1)
            
#             # Get attention weights
#             if model.decoder_rnn_type == 'LSTM':
#                 attn_weights = model.attention(hidden_dec[-1], encoder_outputs)
#             else:
#                 attn_weights = model.attention(hidden_dec[-1], encoder_outputs)
                
#             attentions.append(attn_weights.cpu().numpy())
                
#             # Apply attention
#             attn_weights = attn_weights.unsqueeze(1)
#             context = torch.bmm(attn_weights, encoder_outputs)
            
#             # Combine embedding and context
#             rnn_input = torch.cat((input_emb, context), dim=2)
            
#             # Decoder forward
#             if model.decoder_rnn_type == 'LSTM':
#                 output, (hidden_dec, cell_dec) = model.decoder(rnn_input, (hidden_dec, cell_dec))
#             else:
#                 output, hidden_dec = model.decoder(rnn_input, hidden_dec)
                
#             # Prepare for prediction
#             output = output.squeeze(0)
#             context = context.squeeze(0)
#             input_emb = input_emb.squeeze(0)
            
#             # pred_input = torch.cat((output, context, input_emb.squeeze(0)), dim=1)
#             pred_input = torch.cat((output, context, input_emb), dim=1)
#             prediction = model.fc_out(pred_input)
            
#             # Get predicted token
#             top1 = prediction.argmax(1)
            
#             # Stop if <eos>
#             if top1.item() == output_vocab.eos_idx:
#                 break
                
#             # Add to outputs and use as next input
#             outputs.append(top1.item())
#             input = top1
    
#     return output_vocab.decode(outputs), attentions

# def translate(model, sentence, input_vocab, output_vocab, device, decode_method='greedy', beam_width=5, max_len=50, alpha=1.0):
#     """Translate a sentence using either greedy or beam search decoding"""
#     if decode_method == 'greedy':
#         return greedy_decode(model, sentence, input_vocab, output_vocab, device, max_len)
#     elif decode_method == 'beam':
#         return beam_search_decode(model, sentence, input_vocab, output_vocab, device, beam_width, max_len, alpha)
#     else:
#         raise ValueError(f"Unknown decode method: {decode_method}")







# def translate(model, sentence, input_vocab, output_vocab, device, decode_method='greedy', beam_width=5, max_len=50, length_penalty_alpha=0.75):
#     if decode_method == 'beam':
#         return beam_search_decode(
#             model, sentence, input_vocab, output_vocab, device,
#             beam_width=beam_width, max_len=max_len, length_penalty_alpha=length_penalty_alpha
#         )
#     else:  # greedy search
#         # Your existing greedy search code
#         return greedy_decode(model, sentence, input_vocab, output_vocab, device, max_len)
        # model.eval()

        # # Convert sentence to indices
        # tokens = input_vocab.encode(sentence)
        # src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     embedded = model.embedding(src_tensor)
        #     encoder_outputs, encoder_hidden = model.encoder(embedded)

        #     # Unpack encoder hidden/cell
        #     if getattr(model, "encoder_rnn_type", "LSTM") == 'LSTM':
        #         enc_hidden, enc_cell = encoder_hidden
        #     else:
        #         enc_hidden = encoder_hidden
        #         enc_cell = None

        #     # Prepare initial decoder hidden/cell for all combinations
        #     decoder_num_layers = getattr(model, "decoder_num_layers", 1)
        #     encoder_num_layers = getattr(model, "encoder_num_layers", 1)
        #     hidden_dim = getattr(model, "hidden_dim", enc_hidden.size(-1))
        #     decoder_rnn_type = getattr(model, "decoder_rnn_type", "LSTM")

        #     if decoder_rnn_type == 'LSTM':
        #         # Hidden
        #         if encoder_num_layers < decoder_num_layers:
        #             dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
        #         else:
        #             dec_hidden = enc_hidden[:decoder_num_layers]
        #         # Cell
        #         if enc_cell is not None:
        #             if encoder_num_layers < decoder_num_layers:
        #                 dec_cell = torch.cat([enc_cell] + [enc_cell[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
        #             else:
        #                 dec_cell = enc_cell[:decoder_num_layers]
        #         else:
        #             batch_size = src_tensor.size(0)
        #             dec_cell = torch.zeros(decoder_num_layers, batch_size, hidden_dim, device=src_tensor.device)
        #     else:
        #         if encoder_num_layers < decoder_num_layers:
        #             dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
        #         else:
        #             dec_hidden = enc_hidden[:decoder_num_layers]
        #         dec_cell = None

        #     input = torch.tensor([output_vocab.sos_idx], device=device)
        #     outputs = [output_vocab.sos_idx]
        #     attentions = []

        #     for i in range(max_len):
        #         input_emb = model.decoder_embedding(input).unsqueeze(1)

        #         # Use attention if available
        #         if hasattr(model, "attention"):
        #             attn_weights = model.attention(dec_hidden[-1], encoder_outputs)
        #             attentions.append(attn_weights.cpu().numpy())
        #             attn_weights = attn_weights.unsqueeze(1)
        #             context = torch.bmm(attn_weights, encoder_outputs)
        #             rnn_input = torch.cat((input_emb, context), dim=2)
        #         else:
        #             rnn_input = input_emb

        #         # Decoder forward
        #         if decoder_rnn_type == 'LSTM':
        #             output, (dec_hidden, dec_cell) = model.decoder(rnn_input, (dec_hidden, dec_cell))
        #         else:
        #             output, dec_hidden = model.decoder(rnn_input, dec_hidden)

        #         output = output.squeeze(1)
        #         if hasattr(model, "attention"):
        #             context = context.squeeze(1)
        #             input_emb_squeezed = input_emb.squeeze(1)
        #             pred_input = torch.cat((output, context, input_emb_squeezed), dim=1)
        #         else:
        #             pred_input = output

        #         prediction = model.fc_out(pred_input)
        #         top1 = prediction.argmax(1)
        #         if top1.item() == output_vocab.eos_idx:
        #             break
        #         outputs.append(top1.item())
        #         input = top1

        # return output_vocab.decode(outputs), (attentions if hasattr(model, "attention") else None)


# def plot_confusion_matrix(true_chars, pred_chars, filename="confusion_matrix.png", figsize=(12, 10)):
#     """
#     Generate and save a character-level confusion matrix visualization.
    
#     Args:
#         true_chars: List of true characters
#         pred_chars: List of predicted characters
#         filename: Path to save the output image
#         figsize: Size of the output figure (width, height)
#     """
#     # Get unique characters (sorted for consistent ordering)
#     unique_chars = sorted(list(set(true_chars + pred_chars)))
    
#     # Create label mapping
#     label_indices = {c: i for i, c in enumerate(unique_chars)}
    
#     # Filter only chars present in labels and convert to indices
#     filtered_true = []
#     filtered_pred = []
#     for t, p in zip(true_chars, pred_chars):
#         if t in label_indices and p in label_indices:
#             filtered_true.append(label_indices[t])
#             filtered_pred.append(label_indices[p])
    
#     # Compute confusion matrix
#     cm = confusion_matrix(filtered_true, filtered_pred, labels=range(len(unique_chars)))
    
#     # Create a DataFrame for better visualization
#     cm_df = pd.DataFrame(cm, index=unique_chars, columns=unique_chars)
#     cm_df.index.name = 'True'
#     cm_df.columns.name = 'Predicted'
    
#     # Create figure
#     plt.figure(figsize=figsize)
    
#     # Plot heatmap
#     sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", 
#                 linewidths=0.5, linecolor='gray')
    
#     plt.title('Character-Level Confusion Matrix', fontsize=16)
#     plt.tight_layout()
    
#     # Save figure
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Confusion matrix saved to {filename}")
    
#     return plt.gcf()  # Return the figure if needed for further use


# def plot_detailed_confusion_matrix(true_chars, pred_chars, filename="confusion_matrix_detailed.png"):
#     """
#     Generate a detailed character-level confusion matrix similar to the provided image.
    
#     Args:
#         true_chars: List of true characters
#         pred_chars: List of predicted characters
#         filename: Path to save the output image
#     """
#     # Get unique characters (sorted for consistent ordering)
#     unique_chars = sorted(list(set(true_chars + pred_chars)))
    
#     # Create label mapping
#     label_indices = {c: i for i, c in enumerate(unique_chars)}
    
#     # Filter only chars present in labels and convert to indices
#     filtered_true = []
#     filtered_pred = []
#     for t, p in zip(true_chars, pred_chars):
#         if t in label_indices and p in label_indices:
#             filtered_true.append(label_indices[t])
#             filtered_pred.append(label_indices[p])
    
#     # Compute confusion matrix
#     cm = confusion_matrix(filtered_true, filtered_pred, labels=range(len(unique_chars)))
    
#     # Create a DataFrame for better visualization
#     cm_df = pd.DataFrame(cm, index=unique_chars, columns=unique_chars)
#     cm_df.index.name = 'True'
#     cm_df.columns.name = 'Predicted'
    
#     # Create figure with larger size for detailed view
#     plt.figure(figsize=(20, 18), dpi=300)
    
#     # Plot heatmap with small font size for the values
#     ax = sns.heatmap(cm_df, cmap='Blues', linewidths=0.5, 
#                      annot=True, fmt='d', annot_kws={"size": 5})
    
#     # Set title and labels
#     plt.title('Character-level Confusion Matrix', fontsize=16)
#     plt.ylabel('Ground Truth', fontsize=14)
#     plt.xlabel('Predicted', fontsize=14)
    
#     # Adjust tick label size
#     plt.xticks(fontsize=8, rotation=90)
#     plt.yticks(fontsize=8, rotation=0)
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Save figure
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Confusion matrix saved to {filename}")
    
#     return plt.gcf()  # Return the figure if needed for further use

# def plot_attention_heatmaps(model, test_examples, input_vocab, output_vocab, device, num_examples=9, grid_size=(3, 3)):
#     """
#     Generate and plot attention heatmaps for test examples
    
#     Args:
#         model: The attention-based seq2seq model
#         test_examples: List of input strings to visualize
#         input_vocab: Input vocabulary
#         output_vocab: Output vocabulary
#         device: Device to run the model on
#         num_examples: Number of examples to visualize
#         grid_size: Size of the grid for subplots
#     """
#     fig, axes = plt.subplots(*grid_size, figsize=(15, 15))
#     axes = axes.flatten()
    
#     for i, example in enumerate(test_examples[:num_examples]):
#         # Get prediction and attention weights
#         pred_str, attn_weights = translate(
#             model,
#             example,
#             input_vocab,
#             output_vocab,
#             device,
#             decode_method="greedy",
#             beam_width=5,
#             length_penalty_alpha=1.0
#         )
        
#         # Convert input and output to character lists
#         input_chars = list(example)
#         output_chars = list(pred_str)
        
#         ax = axes[i]
        
#         if attn_weights is not None:
#             # attn_weights = attn_weights[:,len(attn_weights[0]) - len(output_chars) - 1: len(attn_weights[0]) - 1]
#             attn_matrix = np.zeros((len(attn_weights) - 1, len(attn_weights[0][0]) - 1))
#             for i in range(len(attn_matrix)):
#                 for j in range(len(attn_matrix[i])):
#                     # attn_matrix[i][j] = attn_weights[i][0][len(attn_weights[i][0]) - len(output_chars) - 1 + j]
#                     attn_matrix[i][j] = attn_weights[i][0][j + 1]
        
#         # # Check if attention weights are available
#         # if attn_weights is not None:
#         #     # Create attention matrix from weights
#         #     attn_matrix = np.zeros((len(output_chars), len(input_chars)))
            
#         #     # Safely assign attention weights to matrix
#         #     for t, weights in enumerate(attn_weights):
#         #         if t < len(output_chars):
#         #             # Get the minimum length to avoid dimension mismatch
#         #             min_len = min(len(weights), len(input_chars))
#         #             attn_matrix[t, :min_len] = weights[:min_len]
            
#             # Plot heatmap
#             im = ax.imshow(attn_matrix, cmap='Blues')
            
#             # Set labels
#             ax.set_xticks(range(len(input_chars)))
#             ax.set_yticks(range(len(output_chars)))
#             ax.set_xticklabels(input_chars)
#             ax.set_yticklabels(output_chars)
            
#             # Add colorbar
#             plt.colorbar(im, ax=ax)
#         else:
#             # No attention weights available - just show text
#             ax.text(0.5, 0.5, "No attention weights available", 
#                    horizontalalignment='center', verticalalignment='center')
#             ax.axis('off')
        
#         # Add title
#         ax.set_title(f"Input: {example}\nOutput: {pred_str}")
    
#     plt.tight_layout()
#     plt.savefig("attention_heatmaps.png", dpi=300, bbox_inches="tight")
#     plt.close()
    
#     return fig




# # Load a few samples from your predictions file
# def create_transliteration_grid(predictions_file, num_samples=10):
#     # Read predictions
#     samples = []
#     with open(predictions_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         # Select samples (either first few or random)
#         import random
#         selected_lines = random.sample(lines, min(num_samples, len(lines)))
        
#         for line in selected_lines:
#             latin, pred, true = line.strip().split(',')
#             # Check if prediction is correct
#             is_correct = (pred == true)
#             samples.append({
#                 'Latin Input': latin,
#                 'Model Output': pred,
#                 'Ground Truth': true,
#                 'Correct': is_correct
#             })
    
#     # Create DataFrame
#     df = pd.DataFrame(samples)
    
#     # Create figure
#     # fig, ax = plt.figure(figsize=(12, 8), dpi=100)
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
#     # Hide axes
#     ax = plt.subplot(111)
#     ax.axis('off')
    
#     # Create table
#     table = plt.table(
#         cellText=df[['Latin Input', 'Model Output', 'Ground Truth']].values,
#         colLabels=['Latin Input', 'Model Output', 'Ground Truth'],
#         cellLoc='center',
#         loc='center',
#         cellColours=[[('lightgreen' if row['Correct'] else 'mistyrose') for _ in range(3)] for _, row in df.iterrows()]
#     )
    
#     # Style the table
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.scale(1.2, 1.5)
    
#     # Add title
#     plt.title('Hindi Transliteration Results (LSTM Seq2Seq Model)', fontsize=16, pad=20)
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig('transliteration_results.png', bbox_inches='tight', dpi=300)
    
#     return fig, df



# def visualize_attention_for_word(model, input_word, input_vocab, output_vocab, device, 
#                                 save_path=None, figsize=(10, 8)):
#     """
#     Create an attention visualization for a single input word.
    
#     Args:
#         model: The seq2seq model with attention
#         input_word: Input word in Latin script
#         input_vocab: Input vocabulary
#         output_vocab: Output vocabulary
#         device: Device to run the model on
#         save_path: Path to save the visualization
#         figsize: Size of the figure
#     """
#     # Set font that supports Hindi
#     hindi_fonts = [f.name for f in fm.fontManager.ttflist 
#                   if 'Devanagari' in fm.FontProperties(fname=f.fname).get_name()]
#     if hindi_fonts:
#         plt.rcParams['font.family'] = hindi_fonts[0]
#     else:
#         plt.rcParams['font.family'] = 'Nirmala UI'  # Default font that supports Hindi
    
#     # Get prediction and attention weights
#     model.eval()
#     with torch.no_grad():
#         pred_str, attention_weights = translate(
#             model,
#             input_word,
#             input_vocab,
#             output_vocab,
#             device,
#             decode_method="greedy"
#         )
    
#     if attention_weights is None:
#         print("No attention weights available for this model.")
#         return None
    
#     # Process attention weights
#     attention_matrix = []
#     for weights in attention_weights:
#         # Skip the first token (SOS)
#         attention_matrix.append(weights[0][1:])
#     attention_matrix = np.array(attention_matrix)
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Convert input and output to character lists
#     input_chars = list(input_word)
#     output_chars = list(pred_str)
    
#     # Plot heatmap
#     im = ax.imshow(attention_matrix, cmap='Blues')
    
#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('Attention Weight')
    
#     # Set labels
#     ax.set_xticks(range(len(input_chars)))
#     ax.set_yticks(range(len(output_chars)))
#     ax.set_xticklabels(input_chars, fontsize=12)
#     ax.set_yticklabels(output_chars, fontsize=14)
    
#     # Add grid
#     ax.set_xticks(np.arange(-.5, len(input_chars), 1), minor=True)
#     ax.set_yticks(np.arange(-.5, len(output_chars), 1), minor=True)
#     ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
#     # Add title
#     ax.set_title(f"Input: {input_word} → Output: {pred_str}", fontsize=16)
#     ax.set_xlabel("Source (Latin)", fontsize=14)
#     ax.set_ylabel("Target (Hindi)", fontsize=14)
    
#     # Add text annotations
#     for i in range(len(output_chars)):
#         for j in range(len(input_chars)):
#             text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
#                           ha="center", va="center", color="black" if attention_matrix[i, j] < 0.5 else "white")
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
    
#     return fig


# def visualize_attention_custom(input_word, output_word, attention_weights, save_path=None, figsize=(12, 8)):
#     """
#     Create a custom attention visualization where each row shows an output character
#     and the input characters are highlighted based on attention weights.
    
#     Args:
#         input_word: Input word in Latin script
#         output_word: Output word in Hindi script
#         attention_weights: Numpy array of attention weights [output_len, input_len]
#         save_path: Path to save the visualization
#         figsize: Size of the figure
#     """
    
#     # Set font that supports Hindi
#     # hindi_fonts = [f.name for f in fm.fontManager.ttflist 
#     #               if 'Devanagari' in f.FontProperties(fname=f.fname).get_name()]
    
#     # if hindi_fonts:
#     #     plt.rcParams['font.family'] = hindi_fonts[0]
#     # else:
#     #     plt.rcParams['font.family'] = 'Nirmala UI'  # Default font that supports Hindi
#     # Set font that supports Hindi
#     plt.rcParams['font.family'] = 'Nirmala UI'
    
#     # Get dimensions
#     output_len = len(output_word)
#     input_len = len(input_word)
    
#     # Create figure
#     fig, axes = plt.subplots(output_len, 1, figsize=figsize)
#     if output_len == 1:
#         axes = [axes]
    
#     # Title for the entire figure
#     fig.suptitle(f"Input: {input_word} → Output: {output_word}", fontsize=16)
    
#     # For each output character
#     for i, output_char in enumerate(output_word):
#         ax = axes[i]
        
#         # Get attention weights for this output character
#         weights = attention_weights[i]
        
#         # Create a horizontal bar of input characters
#         for j, input_char in enumerate(input_word):
#             # Calculate color intensity based on attention weight
#             color_intensity = weights[j]
            
#             # Create rectangle with color based on attention weight
#             rect = plt.Rectangle((j, 0), 1, 1, 
#                                facecolor=plt.cm.Blues(color_intensity),
#                                edgecolor='black', linewidth=1)
#             ax.add_patch(rect)
            
#             # Add the input character in the center of the rectangle
#             ax.text(j + 0.5, 0.5, input_char, 
#                    ha='center', va='center', 
#                    fontsize=14, 
#                    color='black' if color_intensity < 0.5 else 'white')
            
#             # Add the attention weight below the character
#             # ax.text(j + 0.5, 0.2, f"{weights[j]:.2f}", 
#             #        ha='center', va='center', 
#             #        fontsize=10, 
#             #        color='black' if color_intensity < 0.5 else 'white')
        
#         # Add the output character on the left
#         ax.text(-0.5, 0.5, output_char, ha='center', va='center', fontsize=16)
        
#         # Set axis limits and remove ticks
#         ax.set_xlim(-1, input_len)
#         ax.set_ylim(0, 1)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_frame_on(False)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
    
#     return fig


# def visualize_attention_highlights(input_word, output_word, attention_weights, save_path=None, figsize=(12, 8), highlight_threshold=0.3):
#     """
#     Create a custom attention visualization similar to the sample drawing.
#     Each row shows an output character with the most relevant input characters highlighted.
    
#     Args:
#         input_word: Input word in Latin script
#         output_word: Output word in Hindi script
#         attention_weights: Numpy array of attention weights [output_len, input_len]
#         save_path: Path to save the visualization
#         figsize: Size of the figure
#         highlight_threshold: Threshold for highlighting characters (0-1)
#     """
    
    
#     # Set font that supports Hindi
#     plt.rcParams['font.family'] = 'Nirmala UI'
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Draw input characters at the top
#     for j, char in enumerate(input_word):
#         ax.text(j + 0.5, 0, char, ha='center', va='bottom', fontsize=16)
    
#     # For each output character
#     for i, output_char in enumerate(output_word):
#         y_pos = i + 1  # Position in the y-axis
        
#         # Get attention weights for this output character
#         weights = attention_weights[i]
        
#         # Draw the output character
#         ax.text(-0.5, y_pos, output_char, ha='center', va='center', fontsize=16)
        
#         # Draw rectangles for input characters with significant attention
#         for j, weight in enumerate(weights):
#             # Only highlight characters with attention above threshold
#             if weight > highlight_threshold:
#                 # Calculate color intensity
#                 color_intensity = min(weight, 1.0)  # Cap at 1.0
                
#                 # Create rectangle
#                 rect = Rectangle(
#                     (j, y_pos - 0.4), 1, 0.8,
#                     facecolor=plt.cm.Blues(color_intensity),
#                     edgecolor='black', linewidth=1, alpha=0.8
#                 )
#                 ax.add_patch(rect)
                
#                 # Add the input character in the rectangle
#                 if (j < len(input_word)):
#                     ax.text(j + 0.5, y_pos, input_word[j], 
#                        ha='center', va='center', 
#                        fontsize=14, 
#                        color='black' if color_intensity < 0.5 else 'white')
    
#     # Set axis limits and remove ticks
#     ax.set_xlim(-1, len(input_word) + 0.5)
#     ax.set_ylim(len(output_word) + 0.5, -0.5)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Add title
#     plt.title(f"Input: {input_word}, Output: {output_word}", fontsize=16)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
    
#     return fig


# def visualize_word_attention(model, input_word, input_vocab, output_vocab, device, save_path=None):
#     """
#     Create a custom attention visualization for a single input word.
    
#     Args:
#         model: The seq2seq model with attention
#         input_word: Input word in Latin script
#         input_vocab: Input vocabulary
#         output_vocab: Output vocabulary
#         device: Device to run the model on
#         save_path: Path to save the visualization
#     """
#     # Get prediction and attention weights
#     model.eval()
#     with torch.no_grad():
#         pred_str, attention_weights = translate(
#             model,
#             input_word,
#             input_vocab,
#             output_vocab,
#             device,
#             decode_method="greedy"
#         )
    
#     if attention_weights is None:
#         print("No attention weights available for this model.")
#         return None
    
#     # Process attention weights
#     attention_matrix = []
#     for weights in attention_weights:
#         # Skip the first token (SOS)
#         attention_matrix.append(weights[0][1:])
#     attention_matrix = np.array(attention_matrix)
    
#     # Create the custom visualization
#     return visualize_attention_custom(
#         input_word, 
#         pred_str, 
#         attention_matrix, 
#         save_path=save_path
#     )

#     # return visualize_attention_highlights(
#     #         input_word, 
#     #         pred_str, 
#     #         attention_matrix, 
#     #         save_path=save_path
#     #     )



def main(args):
    # Set up WandB
    if (args.enable_wandb):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model_type": args.model_type,
                "encoder_type": args.encoder_rnn_type,
                "decoder_type": args.decoder_rnn_type,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "encoder_layers": args.encoder_layers,
                "decoder_layers": args.decoder_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "clip": args.clip,
                "language": args.language,
                "teacher_forcing_ratio": args.teacher_forcing_ratio,
                "decode_method": args.decode_method,
                "beam_width": args.beam_width,
                "length_penalty_alpha": args.length_penalty_alpha
            }
        )
    
        # Set a meaningful run name
        wandb.run.name = f"{args.model_type}_{args.language}_emb{args.embedding_dim}_hid{args.hidden_dim}_{args.decode_method}"
    
    # Paths to Dakshina lexicon files
    train_path = f"dakshina_dataset_v1.0/dakshina_dataset_v1.0/{args.language}/lexicons/{args.language}.translit.sampled.train.tsv"
    dev_path = f"dakshina_dataset_v1.0/dakshina_dataset_v1.0/{args.language}/lexicons/{args.language}.translit.sampled.dev.tsv"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading datasets...")
    train_ds = DakshinaLexiconDataset(train_path, max_len=args.max_seq_len)
    input_vocab, output_vocab = train_ds.get_vocabs()
    dev_ds = DakshinaLexiconDataset(dev_path, input_vocab, output_vocab, max_len=args.max_seq_len)
    
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Output vocabulary size: {len(output_vocab)}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(dev_ds)}")

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model based on type
    if args.model_type == "basic":
        model = Seq2SeqRNN(
            input_dim=len(input_vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=len(output_vocab),
            encoder_rnn_type=args.encoder_rnn_type,
            decoder_rnn_type=args.decoder_rnn_type,
            encoder_num_layers=args.encoder_layers,
            decoder_num_layers=args.decoder_layers,
            pad_idx=input_vocab.pad_idx
        ).to(DEVICE)
    elif args.model_type == "attention":
        model = Seq2SeqAttention(
            input_dim=len(input_vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=len(output_vocab),
            encoder_rnn_type=args.encoder_rnn_type,
            decoder_rnn_type=args.decoder_rnn_type,
            encoder_num_layers=args.encoder_layers,
            decoder_num_layers=args.decoder_layers,
            pad_idx=input_vocab.pad_idx,
            dropout=args.dropout
        ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters")
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=input_vocab.pad_idx)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = f"best_{args.model_type}_{args.language}_model.pt"
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, args.clip)
        
        # Evaluate
        val_loss, char_acc, seq_acc = evaluate_combined(
                                        model, 
                                        dev_loader, 
                                        criterion, 
                                        DEVICE, 
                                        input_vocab,
                                        output_vocab, 
                                        decode_method=args.decode_method, 
                                        beam_width=args.beam_width, 
                                        length_penalty_alpha=args.length_penalty_alpha)
        
        # Print progress
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Char Acc: {char_acc:.4f} | Seq Acc: {seq_acc:.4f}")
        
        # Log to WandB
        if (args.enable_wandb):
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "char_accuracy": char_acc,
                "sequence_accuracy": seq_acc
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with val loss: {val_loss:.4f}")
    
    # Load best model and test some examples
    model.load_state_dict(torch.load(best_model_path))
    

    # Path to test set
    test_path = f"dakshina_dataset_v1.0/dakshina_dataset_v1.0/{args.language}/lexicons/{args.language}.translit.sampled.test.tsv"

    test_ds = DakshinaLexiconDataset(test_path, input_vocab, output_vocab, max_len=args.max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if (args.generate_test_predictions):
        test_decode_method = "greedy"
        output_file = "predictions_attention/test_predictions.txt"
        if (args.model_type == "basic"):
            test_decode_method = "beam"
            output_file = "predictions_vanilla/test_predictions.txt"
        # Ensure output directory exists
        os.makedirs("predictions_attention", exist_ok=True)
        

        # For confusion matrix
        all_true_chars = []
        all_pred_chars = []

        # Open file for writing predictions
        with open(output_file, "w", encoding="utf-8") as fout:
            model.eval()
            total_correct = 0
            total = 0
            for src_ids, tgt_ids in tqdm(test_ds, desc="Test Evaluation"):
                # Decode input and target
                src_str = input_vocab.decode(src_ids)
                tgt_str = output_vocab.decode(tgt_ids)
                # Predict using beam search
                pred_str, _ = translate(
                    model,
                    src_str,
                    input_vocab,
                    output_vocab,
                    DEVICE,
                    decode_method=test_decode_method,
                    beam_width=args.beam_width,
                    length_penalty_alpha=args.length_penalty_alpha
                )
                # Write to file: input, predicted, true (comma separated)
                fout.write(f"{src_str},{pred_str},{tgt_str}\n")
                # Sequence-level accuracy
                if pred_str == tgt_str:
                    total_correct += 1
                total += 1
                # For confusion matrix (character level)
                min_len = min(len(pred_str), len(tgt_str))
                all_pred_chars.extend(list(pred_str[:min_len]))
                all_true_chars.extend(list(tgt_str[:min_len]))

            seq_acc = total_correct / total if total > 0 else 0.0
        print(f"\nTest Sequence Accuracy : {seq_acc:.4f}")
        # Build character set from output vocab
        labels = output_vocab.itos[4:]  # skip special tokens
        label_indices = {c: i for i, c in enumerate(labels)}
        # Filter only chars present in labels
        true_idx = [label_indices[c] for c in all_true_chars if c in label_indices]
        pred_idx = [label_indices[c] for c in all_pred_chars if c in label_indices]
        # Compute confusion matrix
        cm = confusion_matrix(true_idx, pred_idx, labels=range(len(labels)))
        print("\nCharacter-level confusion matrix:")
        print("Rows: True, Columns: Predicted")
        print(cm)

    # After collecting all_true_chars and all_pred_chars during evaluation
    # plot_confusion_matrix(all_true_chars, all_pred_chars, "confusion_matrix.png")
    if (args.enable_visualizations):
        plot_detailed_confusion_matrix(all_true_chars, all_pred_chars, "confusion_matrix_attention.png")
        fig, df = create_transliteration_grid('predictions_attention/test_predictions.txt', num_samples=15)

        test_examples = [
            "angrzi", "bengali", "anureet", 
            "afroz", "antarmukh", "amzera", 
            "idol", "anant", "akash"
        ]

        # Load your best model
        model.load_state_dict(torch.load(f"best_{args.model_type}_{args.language}_model.pt"))
        model.eval()

        # Generate heatmaps
        if (args.model_type == "attention"):
            fig = plot_attention_heatmaps(
                model, test_examples, input_vocab, output_vocab, DEVICE
            )
        
            for index, example in enumerate(test_examples):
                visualize_word_attention(model, example, input_vocab, output_vocab, DEVICE, save_path=f"./question6_{index}.png")

    
    
    
    
    # Finish WandB run
    if (args.enable_wandb):
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq Transliteration with Dakshina Dataset")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="DA6401_Assignment_3", 
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                        help="WandB entity name")
    
    # Model architecture
    parser.add_argument("--model_type", type=str, default="attention", choices=["basic", "attention"],
                        help="Type of model to use (basic or attention)")
    parser.add_argument("--encoder_rnn_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"],
                        help="Type of RNN to use for encoder")
    parser.add_argument("--decoder_rnn_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"],
                        help="Type of RNN to use for decoder")
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="Dimension of embedding vectors")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Dimension of hidden states in RNN")
    parser.add_argument("--encoder_layers", type=int, default=3,
                        help="Number of layers in encoder RNN")
    parser.add_argument("--decoder_layers", type=int, default=2,
                        help="Number of layers in decoder RNN")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0007625795353002294,
                        help="Learning rate for optimizer")
    parser.add_argument("--clip", type=float, default=5,
                        help="Gradient clipping value")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1,
                        help="Probability of using teacher forcing during training")
    
    # Decoding parameters
    parser.add_argument("--decode_method", type=str, default="greedy", choices=["greedy", "beam"],
                        help="Decoding method (greedy or beam search)")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width for beam search decoding")
    parser.add_argument("--length_penalty_alpha", type=float, default=1.0,
                        help="Length penalty alpha for beam search (higher values favor longer sequences)")
    parser.add_argument("--compare_decoding", action="store_true",
                        help="Compare both greedy and beam search decoding on test examples")
    
    # Data parameters
    parser.add_argument("--language", type=str, default="hi", 
                        choices=["bn", "hi", "gu", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"],
                        help="Language code for Dakshina dataset")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length")
    
    # Visualizations enable
    parser.add_argument("--enable_visualizations", type=int, default=0,
                        help="If you want to generate the plots, pass this argument as 1")
    
    # Generate test predictions
    parser.add_argument("--generate_test_predictions", type=int, default=0,
                        help="If you want to generate test predictions")
    
    # wandb log enable
    parser.add_argument("--enable_wandb", type=int, default=0,
                        help="If you want to enable wandb logs")
    
    args = parser.parse_args()
    
    main(args)




# batch_size:32
# beam_width:5
# clip:5
# decode_method:"greedy"
# decoder_layers:2
# decoder_type:"LSTM"
# dropout:0.2
# embedding_dim:64
# encoder_layers:3
# encoder_type:"LSTM"
# epochs:15
# hidden_dim:256
# language:"hi"
# learning_rate:0.0007625795353002294
# length_penalty_alpha:1
# model_type:"basic"
# teacher_forcing_ratio:1
