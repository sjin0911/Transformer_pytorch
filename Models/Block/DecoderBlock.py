import torch.nn as nn

from Models.Layer.residual_connection_layer import ResidualConnectionLayer
import copy

class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residual = [ResidualConnectionLayer(copy.deepcopy(norm), dr_rate) for _ in range(3)]
        self.norm = norm


    def forward(self, target, encoder_out, target_mask, src_target_mask):
        out = target
        out = self.residual[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=target_mask))
        out = self.residual[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_target_mask))
        out = self.residual[2](out, self.position_ff)
        return out
    
