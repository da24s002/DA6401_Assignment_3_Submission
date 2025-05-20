from torch.utils.data import Dataset, DataLoader

class SequenceEmbeddings:
    def __init__(self, seqs, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        chars = set(c for seq in seqs for c in seq)
        self.itos = specials + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]
        self.unk_idx = self.stoi["<unk>"]

    def encode(self, seq):
        return [self.sos_idx] + [self.stoi.get(c, self.unk_idx) for c in seq] + [self.eos_idx]
    def decode(self, idxs):
        chars = []
        for i in idxs:
            if i == self.eos_idx:
                break
            if i not in (self.sos_idx, self.pad_idx):
                chars.append(self.itos[i])
        return "".join(chars)
    def __len__(self):
        return len(self.itos)

class DakshinaLexiconDataset(Dataset):
    def __init__(self, tsv_path, input_vocab=None, output_vocab=None, max_len=30):
        # Read (latin, native) pairs
        pairs = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                native, latin = parts[0], parts[1]
                pairs.append((latin, native))
        self.latin_seqs = [p[0] for p in pairs]
        self.native_seqs = [p[1] for p in pairs]
        # Build or use vocabularies
        self.input_vocab = input_vocab or SequenceEmbeddings(self.latin_seqs)
        self.output_vocab = output_vocab or SequenceEmbeddings(self.native_seqs)
        self.max_len = max_len
        self.samples = []
        for src, target in zip(self.latin_seqs, self.native_seqs):
            src_ids = self.input_vocab.encode(src)[:self.max_len]
            target_ids = self.output_vocab.encode(target)[:self.max_len]
            self.samples.append((src_ids, target_ids))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
    def get_vocabs(self):
        return self.input_vocab, self.output_vocab