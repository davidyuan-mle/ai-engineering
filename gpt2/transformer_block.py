import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, configure): 
        super().__init__()
        d_model = configure['d_model']
        context_length = configure['context_length']
        n_heads = configure['n_heads']
        drop_rate = configure['drop_rate']
        qkv_bias = configure['qkv_bias']

        self.attention = MultiHeadAttention(d_model, context_length, n_heads, drop_rate, qkv_bias)
        self.ff = FeedForward(configure)
        self.norm1 = LayerNorm(configure)
        self.norm2 = LayerNorm(configure)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x 
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_out(x)
        x = x + residual 

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_out(x)
        x = x + residual 

        return x 
