import torch
import torch.nn as nn

import random

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # Get attention scores
        attention = self.v(energy).squeeze(2)
        # Apply softmax to get attention weights
        return torch.softmax(attention, dim=1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 encoder_rnn_type='LSTM', decoder_rnn_type='LSTM',
                 encoder_num_layers=1, decoder_num_layers=1, pad_idx=0,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        self.encoder = rnn_cls[encoder_rnn_type](
            embedding_dim, hidden_dim, encoder_num_layers, batch_first=True,
            dropout=dropout if encoder_num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.decoder = rnn_cls[decoder_rnn_type](
            embedding_dim + hidden_dim, hidden_dim, decoder_num_layers, batch_first=True,
            dropout=dropout if decoder_num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * 2 + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.encoder_rnn_type = encoder_rnn_type
        self.decoder_rnn_type = decoder_rnn_type
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.hidden_dim = hidden_dim

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(batch_size, trg_len, self.output_dim, device=src.device)
        embedded = self.embedding(src)
        encoder_outputs, encoder_hidden = self.encoder(embedded)

        # --- Robust hidden/cell state transfer ---
        if self.encoder_rnn_type == 'LSTM':
            enc_hidden, enc_cell = encoder_hidden
        else:
            enc_hidden = encoder_hidden
            enc_cell = None

        if self.decoder_rnn_type == 'LSTM':
            # Hidden state
            if self.encoder_num_layers < self.decoder_num_layers:
                dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (self.decoder_num_layers - self.encoder_num_layers), 0)
            else:
                dec_hidden = enc_hidden[:self.decoder_num_layers]
            # Cell state
            if enc_cell is not None:
                if self.encoder_num_layers < self.decoder_num_layers:
                    dec_cell = torch.cat([enc_cell] + [enc_cell[-1:]] * (self.decoder_num_layers - self.encoder_num_layers), 0)
                else:
                    dec_cell = enc_cell[:self.decoder_num_layers]
            else:
                dec_cell = torch.zeros(self.decoder_num_layers, batch_size, self.hidden_dim, device=src.device)
        else:
            if self.encoder_num_layers < self.decoder_num_layers:
                dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (self.decoder_num_layers - self.encoder_num_layers), 0)
            else:
                dec_hidden = enc_hidden[:self.decoder_num_layers]
            dec_cell = None

        input = trg[:, 0]
        for t in range(1, trg_len):
            input_emb = self.decoder_embedding(input).unsqueeze(1)
            # Always use the top layer's hidden state for attention
            if self.decoder_rnn_type == 'LSTM':
                attn_weights = self.attention(dec_hidden[-1], encoder_outputs)
            else:
                attn_weights = self.attention(dec_hidden[-1], encoder_outputs)
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            rnn_input = torch.cat((input_emb, context), dim=2)
            if self.decoder_rnn_type == 'LSTM':
                output, (dec_hidden, dec_cell) = self.decoder(rnn_input, (dec_hidden, dec_cell))
            else:
                output, dec_hidden = self.decoder(rnn_input, dec_hidden)
            output = output.squeeze(1)
            context = context.squeeze(1)
            input_emb = input_emb.squeeze(1)
            pred_input = torch.cat((output, context, input_emb), dim=1)
            prediction = self.fc_out(pred_input)
            outputs[:, t] = prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs
