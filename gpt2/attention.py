import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, context_length, n_heads, drop_rate=0.1, qkv_bias=False): 
        # assume d_model is divisible by n_heads
        # assume the input dimension d_model is the same as the output dimension
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.context_length = context_length

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_head = self.d_model // self.n_heads
        self.Wq = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.Wk = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.Wv = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.dropout = nn.Dropout(self.drop_rate)

        self.out_proj = nn.Linear(self.d_model, self.d_model) 
        
        self.register_buffer("mask", torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)) # causal mask - the upper triangle are 1 and else are 0

    def forward(self, x):
        B, T, d_model = x.shape # B - batches, T - number of tokens, d_model - input dimension
        queries = self.Wq(x) # (B, T, d_model)
        keys = self.Wk(x)    # (B, T, d_model)
        values = self.Wv(x)  # (B, T, d_model)

        # reshape all Q, K, V to (B, T, n_head, d_head)
        queries = queries.view(B, T, self.n_heads, self.d_head)
        keys = keys.view(B, T, self.n_heads, self.d_head)
        values = values.view(B, T, self.n_heads, self.d_head)

        # tranpose to move n_head to dimension = 1
        queries = queries.transpose(1, 2)    # (B, n_head, T, d_head)
        keys = keys.transpose(1, 2)          # (B, n_head, T, d_head)
        values = values.transpose(1, 2)      # (B, n_head, T, d_head)

        # attention scores Q @ K^T
        attn_scores = queries @ keys.transpose(2, 3)   # (B, n_head, T, T)
        
        # mask boolean 
        mask_bool = self.mask.bool()[:T, :T]  # self.mask has the max context length, while for each input, we only T tokens

        # attention scores masked fill 
        attn_scores.masked_fill_(mask_bool, -torch.inf) # for the upper triangle, set values to -inf 

        # attention weights 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # dropout 
        attn_weights = self.dropout(attn_weights)  # (B, n_head, T, T)

        # context vector 
        context_vec = (attn_weights @ values).transpose(1, 2)  # (B, T, n_head, d_head)

        # flatten
        context_vec = context_vec.contiguous().view(B, T, d_model)  # (B, T, d_model)

        # Linear projection to output 
        output = self.out_proj(context_vec)

        return output