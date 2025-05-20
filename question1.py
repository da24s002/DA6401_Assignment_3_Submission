import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Seq2SeqRNN import Seq2SeqRNN
from VocabUtils import DakshinaLexiconDataset



def collate_fn(batch):
    srcs, targets = zip(*batch)
    src_lens = [len(s) for s in srcs]
    target_lens = [len(t) for t in targets]
    max_src = max(src_lens)
    max_target = max(target_lens)
    pad_srcs = [s + [0]*(max_src-len(s)) for s in srcs]
    pad_targets = [t + [0]*(max_target-len(t)) for t in targets]
    return (torch.tensor(pad_srcs, dtype=torch.long), torch.tensor(pad_targets, dtype=torch.long), src_lens, target_lens)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, target, src_lens, target_lens in dataloader:
        src, target = src.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(src, target)
        # Shift targets to ignore <sos>
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)



def evaluate_combined(model, dataloader, criterion, device, output_vocab):
    model.eval()
    total_loss = 0
    
    # Character accuracy metrics
    total_correct_chars = 0
    total_chars = 0
    
    # Sequence accuracy metrics
    total_correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for src, target, src_lens, target_lens in dataloader:
            src, target = src.to(device), target.to(device)
            batch_size = src.shape[0]
            
            # Forward pass with no teacher forcing
            output = model(src, target, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
            
            # Get predictions
            preds = output.argmax(-1)
            
            # Create mask to ignore padding tokens
            mask = target != output_vocab.pad_idx
            
            # Character accuracy calculation
            correct = (preds == target) & mask
            total_correct_chars += correct[:, 1:].sum().item()
            total_chars += mask[:, 1:].sum().item()
            
            # Sequence accuracy calculation
            for i in range(batch_size):
                # Get actual sequence length (excluding <pad> tokens)
                seq_len = mask[i].sum().item()
                
                # Check if all characters in the sequence match
                # Skip the first token (<sos>) and check only up to sequence length
                sequence_correct = torch.all(preds[i, 1:seq_len] == target[i, 1:seq_len]).item()
                
                if sequence_correct:
                    total_correct_sequences += 1
                    
                total_sequences += 1
    
    # Calculate both accuracy metrics
    char_acc = total_correct_chars / total_chars if total_chars > 0 else 0
    seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0
    
    return total_loss / len(dataloader), char_acc, seq_acc

def main():
    # Paths to Dakshina lexicon files (update with your paths)
    train_path = "dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"


    input_embedding_dim = 64
    hidden_state_dim = 128
    encoder_layers = 2
    decoder_layers = 2
    batch_size = 64
    num_epochs = 10
    encoder_rnn_type = 'LSTM'
    decoder_rnn_type = 'LSTM'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    train_ds = DakshinaLexiconDataset(train_path)
    input_vocab, output_vocab = train_ds.get_vocabs()
    dev_ds = DakshinaLexiconDataset(dev_path, input_vocab, output_vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # Model, optimizer, loss
    model = Seq2SeqRNN(
        input_dim=len(input_vocab),
        embedding_dim=input_embedding_dim,
        hidden_dim=hidden_state_dim,
        output_dim=len(output_vocab),
        encoder_rnn_type=encoder_rnn_type,
        decoder_rnn_type=decoder_rnn_type,
        encoder_num_layers=encoder_layers,
        decoder_num_layers=decoder_layers,
        pad_idx=input_vocab.pad_idx
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab.pad_idx)
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        # val_loss, val_acc = evaluate(model, dev_loader, criterion, DEVICE, output_vocab)
        # print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Char Acc={val_acc:.4f}")
        val_loss, char_acc, seq_acc = evaluate_combined(model, dev_loader, criterion, DEVICE, output_vocab)
        print(f"Validation Loss: {val_loss:.4f} | Char Acc: {char_acc:.4f} | Seq Acc: {seq_acc:.4f}")
        

if __name__ == "__main__":
    main()
