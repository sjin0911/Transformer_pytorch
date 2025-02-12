import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0)/d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        pos_embed = self.encoding[:, :seq_len, :]

        # print(f"x shape: {x.shape}, pos_embed shape: {pos_embed.shape}")
        out = x + pos_embed
        return out