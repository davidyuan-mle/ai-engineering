import torch
from pooling import SumPooling, MeanPooling, AttentionPooling, DIN

torch.manual_seed(123)

B, T, d, K = 4, 32, 16, 4  # batch, history length, embedding dim, number of attention queries

seq = torch.randn(B, T, d)  # (B, T, d) - random behavior embeddings (e.g. clicked ad topics)
target = torch.randn(B, d)  # (B, d) - random candidate ad embedding

# random valid lengths per row, to build a padding mask
lengths = torch.randint(1, T + 1, (B,))  # (B,) - each row's number of real (unpadded) behaviors
mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T), 1.0 = valid, 0.0 = padding

print(f"seq: {tuple(seq.shape)}, target: {tuple(target.shape)}, mask: {tuple(mask.shape)}")
print(f"valid lengths per row: {lengths.tolist()}")
print()

sum_pool = SumPooling()
mean_pool = MeanPooling()
attn_pool = AttentionPooling(d=d, k=K)
din = DIN(d=d)

sum_out = sum_pool(seq, mask)
mean_out = mean_pool(seq, mask)
attn_out = attn_pool(seq, mask)
din_out, din_weights = din(seq, target, mask)

print(f"Sum pooling output:        {tuple(sum_out.shape)}")
print(f"Mean pooling output:       {tuple(mean_out.shape)}")
print(f"Attention pooling output:  {tuple(attn_out.shape)}")
print(f"DIN pooled output:         {tuple(din_out.shape)}   (weights: {tuple(din_weights.shape)})")
print()

# Sanity check: mean pooling should ignore whatever garbage sits in the padded tail.
seq_dirty = seq.clone()
seq_dirty[~mask.bool()] = 999.0
mean_out_dirty = mean_pool(seq_dirty, mask)
print(f"Mean pooling unaffected by garbage padding: {torch.allclose(mean_out, mean_out_dirty, atol=1e-5)}")

# Sanity check: DIN's weights are NOT softmax-normalized -- they need not sum to 1 over T.
print(f"DIN weight sums per row (not expected to be 1.0): {[round(v, 3) for v in din_weights.sum(dim=1).tolist()]}")
