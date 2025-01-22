import torch
import torch.nn as nn

from Models.Block.DecoderBlock import DecoderBlock
from Models.Block.EncoderBlock import EncoderBlock
from Models.Embedding.PositionalEncoding import PositionalEncoding
from Models.Embedding.TokenEmbedding import TokenEmbedding
from Models.Embedding.TransformerEmbedding import TransformerEmbedding
from Models.Layer.multi_head_attention_layer import MultiHeadAttentionLayer
from Models.Layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from Models.Model.Decoder import Decoder
from Models.Model.Encoder import Encoder
from Models.Model.Transformer import Transformer


def build_model(src_vocab_size, 
                target_vocab_size, 
                device=torch.device('cpu'), 
                max_len=256, 
                d_embed=512, 
                n_layer=6, 
                d_model=512, 
                h=8, 
                d_ff=2048,
                dr_rate = 0.1):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
        d_embed = d_embed,
        vocab_size = src_vocab_size
    )

    target_token_embed = TokenEmbedding(
        d_embed = d_embed,
        vocab_size = target_vocab_size
    )

    pos_embed = PositionalEncoding(
        d_embed = d_embed,
        max_len = max_len,
        device = device
    )

    src_embed = TransformerEmbedding(
        token_embed = src_token_embed,
        pos_embed = copy(pos_embed),
        dr_rate = dr_rate
    )

    target_embed = TransformerEmbedding(
        token_embed = target_token_embed,
        pos_embed = copy(pos_embed),
        dr_rate = dr_rate
    )

    attention = MultiHeadAttentionLayer(
        d_model = d_model,
        h = h,
        qkv_fc = nn.Linear(d_embed, d_model), 
        out_fc = nn.Linear(d_model, d_embed),
        dr_rate = dr_rate
    )

    position_ff = PositionWiseFeedForwardLayer(
        fc1 = nn.Linear(d_embed, d_ff),
        fc2 = nn.Linear(d_ff, d_embed),
        dr_rate = dr_rate
    )
    norm = nn.LayerNorm(d_embed)

    encoder_block = EncoderBlock(
        self_attention = copy(attention),
        position_ff = copy(position_ff),
        norm = copy(norm),
        dr_rate = dr_rate
    )
    
    decoder_block = DecoderBlock(
        self_attention = copy(attention),
        cross_attention = copy(attention),
        position_ff = copy(position_ff),
        norm = copy(norm),
        dr_rate = dr_rate
    )

    encoder = Encoder(
        encoder_block = encoder_block,
        n_layer = n_layer,
        norm = copy(norm)
    )
    
    decoder = Decoder(
        decoder_block = decoder_block,
        n_layer = n_layer,
        norm = copy(norm)
    )

    generator = nn.Linear(d_model, target_vocab_size)

    model = Transformer(
        src_embed = src_embed,
        target_embed = target_embed,
        encoder = encoder,
        decoder = decoder,
        generator = generator.to(device)
    )

    model.device = device

    return model