import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class Transformer(nn.Module):
    
    def __init__(self, src_embed, target_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        # src: input sentence
        # out: context
        out = self.encoder(self.src_embed(src), src_mask) 
        return out
    
    def decode(self, target, encoder_out, target_mask, src_target_mask):
        # target: sentence
        # encoder_out: context
        out = self.decoder(self.target_embed(target), encoder_out, target_mask, src_target_mask)
        return out
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        src_target_mask = self.make_src_target_mask(src, target)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(target, encoder_out, target_mask, src_target_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out
    
    def calculate_loss(self, out_pos, out_neg, pos, neg):
        pos_loss = F.cross_entropy(out_pos, pos)
        neg_loss = F.cross_entropy(out_neg, neg)

        total_loss = pos_loss + neg_loss
        return total_loss

    def make_pad_mask(self, query, key, pad_idx=-1):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(-1), key.size(-1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grid = False
        return mask
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask
    
    def make_subsequent_mask(self, query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False)

        return mask

    def make_target_mask(self, target):
        pad_mask = self.make_pad_mask(target, target)
        seq_mask = self.make_subsequent_mask(target, target)
        mask = pad_mask & seq_mask
        return mask
    
    def make_src_target_mask(self, src, target):
        pad_mask = self.make_pad_mask(target, src)
        return pad_mask