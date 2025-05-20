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
