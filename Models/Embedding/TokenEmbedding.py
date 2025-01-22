import torch.nn as nn
import math

class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size, padding_idx=0):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=padding_idx)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out
