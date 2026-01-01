import torch
from utils import GELU, LayerNorm, FeedForward
from attention import MultiHeadAttention
from transformer_block import TransformerBlock
from gpt2model import GPT2Model

configure = {
    'vocabulary': 50257,      # Vocabulary size
    'd_model': 768,           # Embedding dimension
    'context_length': 1024,   # Max context length 
    'n_heads': 12,             # Number of attention heads
    'n_layers': 12,           # Number of transformer layers 
    'drop_rate': 0.1,         # Dropout rate
    'qkv_bias': False         # QKV bias
}

torch.manual_seed(123)

# input tokens 
batch = torch.tensor([
    [6109, 3626, 6100, 345], 
    [6109, 1110, 6622, 257]
    ])

model = GPT2Model(configure)
output = model(batch)

print(f"Input shape: {batch.shape}")
print(f"Output shape: {output.shape}")
print(output)


# number of parameters 
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")




# torch.manual_seed(789)
# x = torch.randn(2, 3)
# gelu = GELU()
# layer_norm = LayerNorm(configure)
# feed_forward = FeedForward(configure)

# print(f"GELU: \n {gelu(x)}")
# print(f"Layer Norm: \n {layer_norm(x)}")
# print(f"Feed Forward: \n {feed_forward(x)}")

# print(torch.triu(torch.ones(3, 3), diagonal=1))

# transformer = TransformerBlock(configure)
# x = torch.randn(2, 3, 4)
# y = transformer(x)
# print(y)


