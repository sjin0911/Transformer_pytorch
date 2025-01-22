import torch
from torch.utils.data import Dataset

class DummyTextDataset(Dataset):
    
    def __init__(self, vocab_size=50, seq_len=256, num_samples=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        self.data = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]