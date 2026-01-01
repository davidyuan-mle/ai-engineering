import torch.nn as nn
import torch

# GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, configure):  # configure is a dictionary containing the model configuration
        super().__init__()
        self.scale = nn.Parameter(torch.ones(configure['d_model']))
        self.shift = nn.Parameter(torch.zeros(configure['d_model']))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, configure): # configure is a dictionary containing the model configuration
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(configure['d_model'], 4 * configure['d_model']),
            GELU(),
            nn.Linear(4 * configure['d_model'], configure['d_model']),
        )

    def forward(self, x):
        return self.layers(x)

