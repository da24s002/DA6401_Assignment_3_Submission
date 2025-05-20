import torch
import torch.nn as nn
import random

class Seq2SeqRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, 
                 encoder_rnn_type='LSTM', decoder_rnn_type='LSTM', 
                 encoder_num_layers=1, decoder_num_layers=1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        self.encoder = rnn_cls[encoder_rnn_type](embedding_dim, hidden_dim, encoder_num_layers, batch_first=True)
        self.decoder_embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        self.decoder = rnn_cls[decoder_rnn_type](embedding_dim, hidden_dim, decoder_num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.encoder_rnn_type = encoder_rnn_type
        self.decoder_rnn_type = decoder_rnn_type
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers

    def forward(self, src, target, teacher_forcing_ratio=0.5):
        batch_size, src_len = src.shape
        batch_size, target_len = target.shape
        outputs = torch.zeros(batch_size, target_len, self.output_dim, device=src.device)
        embedded = self.embedding(src)
        encoder_outputs, encoder_hidden = self.encoder(embedded)

        # --- Handle hidden/cell state transfer between encoder and decoder ---
        # LSTM returns tuple, others return tensor
        if self.encoder_rnn_type == 'LSTM':
            enc_hidden, enc_cell = encoder_hidden
        else:
            enc_hidden = encoder_hidden
            enc_cell = None

        # Prepare initial decoder states
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
                # Encoder is not LSTM: cell state is zeros
                dec_cell = torch.zeros(self.decoder_num_layers, batch_size, dec_hidden.size(-1), device=src.device)
        else:
            # Decoder is RNN/GRU, only hidden state needed
            if self.encoder_num_layers < self.decoder_num_layers:
                dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (self.decoder_num_layers - self.encoder_num_layers), 0)
            else:
                dec_hidden = enc_hidden[:self.decoder_num_layers]
            dec_cell = None

        # --- Decoder loop ---
        input = target[:, 0]  # <sos>
        for t in range(1, target_len):
            input_emb = self.decoder_embedding(input).unsqueeze(1)
            if self.decoder_rnn_type == 'LSTM':
                output, (dec_hidden, dec_cell) = self.decoder(input_emb, (dec_hidden, dec_cell))
            else:
                output, dec_hidden = self.decoder(input_emb, dec_hidden)
            pred = self.fc_out(output.squeeze(1))
            outputs[:, t] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = target[:, t] if teacher_force else top1
        return outputs
