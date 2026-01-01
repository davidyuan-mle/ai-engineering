import torch
import torch.nn as nn 
from utils import LayerNorm
from transformer_block import TransformerBlock 

class GPT2Model(nn.Module):
    def __init__(self, configure):
        super().__init__()
        self.token_embedding = nn.Embedding(configure['vocabulary'], configure['d_model']) 
        self.positional_embedding = nn.Embedding(configure['context_length'], configure['d_model'])
        self.dropout = nn.Dropout(configure['drop_rate'])

        self.transformer_blocks = nn.Sequential(
            * [TransformerBlock(configure) for _ in range(configure['n_layers'])]
        )

        self.final_norm = LayerNorm(configure) 
        self.out_head = nn.Linear(configure['d_model'], configure['vocabulary'], bias=False)

    def forward(self, input_tokens):
        B, T = input_tokens.shape 
        tok_embeds = self.token_embedding(input_tokens)
        #pos_embeds = self.positional_embedding(torch.arange(T), device=input_tokens.device)
        pos_embeds = self.positional_embedding(torch.arange(T))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x) 
        logits = self.out_head(x)

        return logits

