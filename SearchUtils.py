import torch
from Seq2SeqRNN import Seq2SeqRNN
from Seq2SeqAttention import Seq2SeqAttention
from queue import PriorityQueue


class BeamSearchNode:
    def __init__(self, hidden_state, previous_node, word_id, log_prob, length):
        self.h = hidden_state  # Hidden state (can be a tuple for LSTM)
        self.prev_node = previous_node  # Previous node in the sequence
        self.word_id = word_id  # Current word ID
        self.logp = log_prob  # Log probability
        self.length = length  # Length of sequence
        
    # def eval(self, alpha=1.0):
    #     # Length normalization to avoid favoring short sequences
    #     return self.logp / float(self.length - 1 + 1e-6) + alpha * 0
    
    # def __lt__(self, other):
    #     # For priority queue comparison
    #     return self.length < other.length
    def __lt__(self, other):
        # Higher score = higher priority
        return self.eval() > other.eval()

    def eval(self, alpha=1.0):
        # Length normalization with proper alpha
        return self.logp / (float(self.length)**alpha) if self.length > 0 else self.logp



def greedy_decode(model, sentence, input_vocab, output_vocab, device, max_len=50):
    model.eval()

    # Convert sentence to indices
    tokens = input_vocab.encode(sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        embedded = model.embedding(src_tensor)
        encoder_outputs, encoder_hidden = model.encoder(embedded)

        # Unpack encoder hidden/cell
        if getattr(model, "encoder_rnn_type", "LSTM") == 'LSTM':
            enc_hidden, enc_cell = encoder_hidden
        else:
            enc_hidden = encoder_hidden
            enc_cell = None

        # Prepare initial decoder hidden/cell for all combinations
        decoder_num_layers = getattr(model, "decoder_num_layers", 1)
        encoder_num_layers = getattr(model, "encoder_num_layers", 1)
        hidden_dim = getattr(model, "hidden_dim", enc_hidden.size(-1))
        decoder_rnn_type = getattr(model, "decoder_rnn_type", "LSTM")

        if decoder_rnn_type == 'LSTM':
            # Hidden
            if encoder_num_layers < decoder_num_layers:
                dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
            else:
                dec_hidden = enc_hidden[:decoder_num_layers]
            # Cell
            if enc_cell is not None:
                if encoder_num_layers < decoder_num_layers:
                    dec_cell = torch.cat([enc_cell] + [enc_cell[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
                else:
                    dec_cell = enc_cell[:decoder_num_layers]
            else:
                batch_size = src_tensor.size(0)
                dec_cell = torch.zeros(decoder_num_layers, batch_size, hidden_dim, device=src_tensor.device)
        else:
            if encoder_num_layers < decoder_num_layers:
                dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (decoder_num_layers - encoder_num_layers), 0)
            else:
                dec_hidden = enc_hidden[:decoder_num_layers]
            dec_cell = None

        input = torch.tensor([output_vocab.sos_idx], device=device)
        outputs = [output_vocab.sos_idx]
        attentions = []

        for i in range(max_len):
            input_emb = model.decoder_embedding(input).unsqueeze(1)

            # Use attention if available
            if hasattr(model, "attention"):
                attn_weights = model.attention(dec_hidden[-1], encoder_outputs)
                attentions.append(attn_weights.cpu().numpy())
                attn_weights = attn_weights.unsqueeze(1)
                context = torch.bmm(attn_weights, encoder_outputs)
                rnn_input = torch.cat((input_emb, context), dim=2)
            else:
                rnn_input = input_emb

            # Decoder forward
            if decoder_rnn_type == 'LSTM':
                output, (dec_hidden, dec_cell) = model.decoder(rnn_input, (dec_hidden, dec_cell))
            else:
                output, dec_hidden = model.decoder(rnn_input, dec_hidden)

            output = output.squeeze(1)
            if hasattr(model, "attention"):
                context = context.squeeze(1)
                input_emb_squeezed = input_emb.squeeze(1)
                pred_input = torch.cat((output, context, input_emb_squeezed), dim=1)
            else:
                pred_input = output

            prediction = model.fc_out(pred_input)
            top1 = prediction.argmax(1)
            if top1.item() == output_vocab.eos_idx:
                break
            outputs.append(top1.item())
            input = top1

    return output_vocab.decode(outputs), (attentions if hasattr(model, "attention") else None)

def beam_search_decode(model, sentence, input_vocab, output_vocab, device, beam_width=5, max_len=50, length_penalty_alpha=0.75):
    """Beam search decoding for sequence generation, robust to both vanilla and attention models"""
    model.eval()
    
    # Convert sentence to indices
    tokens = input_vocab.encode(sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encoder forward pass
        embedded = model.embedding(src_tensor)
        encoder_outputs, encoder_hidden = model.encoder(embedded)
        
        # Handle different encoder/decoder layer counts
        if model.encoder_rnn_type == 'LSTM':
            enc_hidden, enc_cell = encoder_hidden
            
            # Adjust hidden state for decoder layers
            if model.encoder_num_layers != model.decoder_num_layers:
                if model.encoder_num_layers < model.decoder_num_layers:
                    dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (model.decoder_num_layers - model.encoder_num_layers), 0)
                    dec_cell = torch.cat([enc_cell] + [enc_cell[-1:]] * (model.decoder_num_layers - model.encoder_num_layers), 0)
                else:
                    dec_hidden = enc_hidden[:model.decoder_num_layers]
                    dec_cell = enc_cell[:model.decoder_num_layers]
            else:
                dec_hidden = enc_hidden
                dec_cell = enc_cell
            
            encoder_hidden = (dec_hidden, dec_cell)
        else:
            # For RNN/GRU
            enc_hidden = encoder_hidden
            
            # Adjust hidden state for decoder layers
            if model.encoder_num_layers != model.decoder_num_layers:
                if model.encoder_num_layers < model.decoder_num_layers:
                    dec_hidden = torch.cat([enc_hidden] + [enc_hidden[-1:]] * (model.decoder_num_layers - model.encoder_num_layers), 0)
                else:
                    dec_hidden = enc_hidden[:model.decoder_num_layers]
            else:
                dec_hidden = enc_hidden
            
            encoder_hidden = dec_hidden
        
        # Start with <sos> token
        sos_token = torch.tensor([output_vocab.sos_idx], device=device)
        
        # Initialize beams
        if model.decoder_rnn_type == 'LSTM':
            hidden, cell = encoder_hidden
            start_node = BeamSearchNode(
                hidden_state=(hidden, cell),
                previous_node=None,
                word_id=sos_token,
                log_prob=0.0,
                length=1
            )
        else:
            start_node = BeamSearchNode(
                hidden_state=encoder_hidden,
                previous_node=None,
                word_id=sos_token,
                log_prob=0.0,
                length=1
            )
        
        # Priority queue for beam search
        nodes = PriorityQueue()
        nodes.put((-start_node.eval(length_penalty_alpha), start_node))
        end_nodes = []
        
        # Beam search iterations
        for _ in range(max_len):
            if nodes.qsize() == 0 or len(end_nodes) >= beam_width:
                break
                
            # Get current best node
            score, n = nodes.get()
            
            # If we reached EOS token, add to end_nodes
            if n.word_id.item() == output_vocab.eos_idx and n.prev_node is not None:
                end_nodes.append((score, n))
                continue
                
            # Decode one step
            decoder_input = n.word_id
            
            # Get hidden state from the node
            if model.decoder_rnn_type == 'LSTM':
                hidden_dec, cell_dec = n.h
            else:
                hidden_dec = n.h
                
            # Check if we're using attention or vanilla model
            has_attention = hasattr(model, "attention")
            
            # Prepare input embedding - only one unsqueeze to avoid 4D tensor
            input_emb = model.decoder_embedding(decoder_input).unsqueeze(1)
            
            # Forward pass through decoder
            if has_attention:
                # Attention model
                # Get attention weights - handle different dimensions properly
                if model.decoder_rnn_type == 'LSTM':
                    # Get the last layer's hidden state and ensure it has the right shape
                    h = hidden_dec[-1]
                    # Make sure h has shape [batch_size, hidden_dim]
                    if h.dim() == 3:
                        h = h.squeeze(0)
                    attn_weights = model.attention(h, encoder_outputs)
                else:
                    h = hidden_dec[-1]
                    if h.dim() == 3:
                        h = h.squeeze(0)
                    attn_weights = model.attention(h, encoder_outputs)
                
                attn_weights = attn_weights.unsqueeze(1)
                context = torch.bmm(attn_weights, encoder_outputs)
                rnn_input = torch.cat((input_emb, context), dim=2)
                
                # Decoder forward pass
                if model.decoder_rnn_type == 'LSTM':
                    output, (hidden_dec, cell_dec) = model.decoder(rnn_input, (hidden_dec, cell_dec))
                    decoder_hidden = (hidden_dec, cell_dec)
                else:
                    output, hidden_dec = model.decoder(rnn_input, hidden_dec)
                    decoder_hidden = hidden_dec
                    
                # Prepare for prediction
                output = output.squeeze(1)
                context = context.squeeze(1)
                input_emb = input_emb.squeeze(1)
                pred_input = torch.cat((output, context, input_emb), dim=1)
                prediction = model.fc_out(pred_input)
            else:
                # Vanilla model
                if model.decoder_rnn_type == 'LSTM':
                    output, (hidden_dec, cell_dec) = model.decoder(input_emb, (hidden_dec, cell_dec))
                    decoder_hidden = (hidden_dec, cell_dec)
                else:
                    output, hidden_dec = model.decoder(input_emb, hidden_dec)
                    decoder_hidden = hidden_dec
                    
                prediction = model.fc_out(output.squeeze(1))
            
            # Get top-k tokens
            log_probs = torch.log_softmax(prediction, dim=1)
            top_log_probs, top_indices = log_probs.topk(beam_width)
            
            # Add new candidates to beam
            for k in range(beam_width):
                token_idx = top_indices[0, k].item()
                token_log_prob = top_log_probs[0, k].item()
                new_log_prob = n.logp + token_log_prob
                
                # Create new beam node
                node = BeamSearchNode(
                    hidden_state=decoder_hidden,
                    previous_node=n,
                    word_id=torch.tensor([token_idx], device=device),
                    log_prob=new_log_prob,
                    length=n.length + 1
                )
                
                # Add to priority queue
                nodes.put((-node.eval(length_penalty_alpha), node))
        
        # If no complete sequences, use the best incomplete ones
        if len(end_nodes) == 0:
            for _ in range(min(beam_width, nodes.qsize())):
                if not nodes.empty():
                    end_nodes.append(nodes.get())
        
        # Get the best sequence
        if end_nodes:
            end_nodes = sorted(end_nodes, key=lambda x: x[0])
            best_node = end_nodes[0][1]
        else:
            return output_vocab.decode([output_vocab.sos_idx]), None
        
        # Backtrack to get the sequence
        sequence = []
        current_node = best_node
        while current_node.prev_node is not None:
            sequence.append(current_node.word_id.item())
            current_node = current_node.prev_node
        
        # Add <sos> token and reverse
        sequence.append(output_vocab.sos_idx)
        sequence = sequence[::-1]
        
        return output_vocab.decode(sequence), None
    

def translate(model, sentence, input_vocab, output_vocab, device, decode_method='greedy', beam_width=5, max_len=50, length_penalty_alpha=0.75):
    if decode_method == 'beam':
        return beam_search_decode(
            model, sentence, input_vocab, output_vocab, device,
            beam_width=beam_width, max_len=max_len, length_penalty_alpha=length_penalty_alpha
        )
    else:  # greedy search
        # Your existing greedy search code
        return greedy_decode(model, sentence, input_vocab, output_vocab, device, max_len)