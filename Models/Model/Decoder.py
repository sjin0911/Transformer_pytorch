import torch.nn as nn
import copy

class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, target, encoder_out, target_mask, src_target_mask):
        out = target
        for layer in self.layers:
            out = layer(out, encoder_out, target_mask, src_target_mask)
        out = self.norm(out)
        return out