import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import copy

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.copy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.copy(qkv_fc)
        self.v_fc = copy.copy(qkv_fc)
        self.out_fc = out_fc # (d_model, d_embed)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, seq_len, d_k)

        n_batch = query.shape[0]

        def transform(x, fc):
            out = fc(x) # out: (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_embed)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_embed)
            return out
        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query=query, key=key, value=value, mask=mask)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)

        return out
            

    def calculate_attention(self, query, key, value, mask=None):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, 1, seq_len, seq_len)

        d_k = key.shape[-1]
        #attention_score: (n_batch, h, seq_len, seq_len)
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        #out: (n_batch, h, seq_len, d_k)
        out = torch.matmul(attention_prob, value)
        return out