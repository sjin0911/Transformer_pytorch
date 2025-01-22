import torch.nn as nn
import copy

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.layers = []
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm(out)
        return out