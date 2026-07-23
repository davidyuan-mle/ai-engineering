import torch.nn as nn
import torch

# Sum pooling - naive baseline. Magnitude grows with history length.
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq, mask=None):
        B, T, d = seq.shape  # B - batch, T - history length, d - embedding dim
        if mask is not None:
            seq = seq * mask.unsqueeze(-1)  # (B, T, d) * (B, T, 1) -> (B, T, d), zero out padded positions
        pooled = seq.sum(dim=1)  # sum over T -> (B, d)
        return pooled


# Mean pooling - baseline. Fixes the magnitude issue, but weighs every past event equally.
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq, mask=None):
        B, T, d = seq.shape  # B - batch, T - history length, d - embedding dim
        if mask is None:
            mask = torch.ones(B, T, device=seq.device, dtype=seq.dtype)  # (B, T), everything valid
        masked = seq * mask.unsqueeze(-1)  # (B, T, d), zero out padded positions
        summed = masked.sum(dim=1)  # sum over T -> (B, d)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1) unmasked length per row, avoid div-by-0
        pooled = summed / lengths  # (B, d)
        return pooled


# Simple attention pooling - K learned query vectors, shared across users/candidates.
# Candidate-agnostic: doesn't know which ad it's scoring.
class AttentionPooling(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d = d
        self.k = k
        self.q = nn.Parameter(torch.randn(k, d) * 0.02)  # learned query vectors (K, d), shared across the batch

    def forward(self, seq, mask=None):
        B, T, d = seq.shape  # B - batch, T - history length, d - embedding dim

        scores = seq @ self.q.transpose(0, 1) / (d ** 0.5)  # (B, T, d) @ (d, K) -> (B, T, K)

        if mask is not None:
            valid = mask.bool().unsqueeze(-1).expand(B, T, self.k)  # (B, T, K)
            scores = scores.masked_fill(~valid, float("-inf"))  # padded positions -> -inf before softmax

        weights = torch.softmax(scores, dim=1)  # softmax over T -> (B, T, K), each query's weights sum to 1 over T

        pooled = weights.transpose(1, 2) @ seq  # (B, K, T) @ (B, T, d) -> (B, K, d)
        pooled = pooled.mean(dim=1)  # mean over K queries -> (B, d)
        return pooled


# Dice activation (DIN paper) - PReLU whose kink point adapts to each channel's
# running mean/variance instead of sitting fixed at 0.
class Dice(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, affine=False)  # tracks per-channel running mean/var, no learned scale/shift
        self.alpha = nn.Parameter(torch.zeros(dim))  # learned negative-side slope, like PReLU's alpha

    def forward(self, x):
        shape = x.shape  # (..., dim) - e.g. (B, T, dim) or (B, dim)
        x_flat = x.reshape(-1, shape[-1])  # (N, dim), flatten leading dims for BatchNorm1d

        x_norm = self.bn(x_flat)  # (x - E[x]) / sqrt(Var[x] + eps) per channel -> (N, dim)
        p = torch.sigmoid(x_norm)  # data-adaptive gate in (0,1), replaces PReLU's hard 0/1 switch at x=0 -> (N, dim)

        out = p * x_flat + (1 - p) * self.alpha * x_flat  # p->1: identity; p->0: alpha * x (learned negative slope)
        return out.reshape(shape)  # back to original shape


# DIN (Deep Interest Network, Alibaba 2018) - target-aware pooling. Each past
# behavior is scored against the candidate through a small activation MLP, then
# weight-summed WITHOUT softmax: the sum of weights itself carries a signal
# (how much of the history is relevant), which softmax would normalize away.
class DIN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(4 * d, 2 * d)  # concat[s, t, s*t, s-t] -> (4d -> 2d)
        self.act1 = nn.PReLU()  # single learnable negative slope, shared across channels
        self.fc2 = nn.Linear(2 * d, d)  # (2d -> d)
        self.act2 = nn.PReLU()  # single learnable negative slope, shared across channels
        self.fc3 = nn.Linear(d, 1)  # (d -> 1), raw relevance score, no final activation

    def forward(self, seq, target, mask=None):
        B, T, d = seq.shape  # B - batch, T - history length, d - embedding dim

        s = seq  # (B, T, d) - each past behavior
        t = target.unsqueeze(1).expand(B, T, d)  # (B, d) -> (B, 1, d) -> (B, T, d), broadcast candidate to every position

        interaction = torch.cat([s, t, s * t, s - t], dim=-1)  # (B, T, 4d), explicit relevance features fed to the MLP

        h = self.act1(self.fc1(interaction))  # (B, T, 4d) -> (B, T, 2d)
        h = self.act2(self.fc2(h))  # (B, T, 2d) -> (B, T, d)
        scores = self.fc3(h).squeeze(-1)  # (B, T, d) -> (B, T, 1) -> (B, T), raw un-normalized relevance score

        if mask is not None:
            scores = scores * mask  # (B, T), zero out padded positions -- no softmax follows, so a plain multiply suffices

        weights = scores.unsqueeze(-1)  # (B, T, 1)
        pooled = (seq * weights).sum(dim=1)  # weighted sum over T -> (B, d); magnitude reflects total activated interest

        return pooled, scores  # scores: (B, T) unnormalized per-behavior weights, useful for inspection
