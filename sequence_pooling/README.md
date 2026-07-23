PyTorch implementations of sum, mean, learned-query attention, and DIN (target-aware) pooling for a user behavior sequence, feeding a downstream cross-network like DCNv2.

## Sum pooling

Adds up every behavior embedding. Cheap, but the pooled vector's magnitude scales with history length — a user with 30 clicks looks very different from one with 3, even if their taste is identical.

`(B, T, d)` → sum over `T` → `(B, d)`

## Mean pooling

Divides by the (unmasked) sequence length. Fixes the magnitude problem, but every past event gets equal weight — an old click counts the same as yesterday's, regardless of which candidate is being scored.

`(B, T, d)` → mean over `T` → `(B, d)`

## Attention pooling

Introduces `K` learned query vectors and lets the sequence "vote" on each one via scaled dot-product attention. The queries are shared across users and candidates, so the model learns `K` generic interest modes — but the pooling still doesn't know which ad it's scoring.

`(B, T, d) @ (d, K) / sqrt(d)` → scores `(B, T, K)` → softmax over `T` → weights `(B, T, K)` → weighted sum → `(B, K, d)` → mean over `K` → `(B, d)`

## DIN (Deep Interest Network)

Target-aware pooling. Each past behavior is scored against the *candidate* through a small activation MLP fed `[s, t, s⊙t, s−t]`, then weight-summed **without softmax** — the sum of weights itself carries a signal (how much of the history is relevant to this candidate) that softmax would normalize away. Padded positions are zeroed out directly rather than set to `-inf`, since there's no softmax to follow.

`(B, T, d)`, `(B, d)` → concat `[s, t, s*t, s-t]` → `(B, T, 4d)` → activation MLP → scores `(B, T)` → weighted sum → `(B, d)`

### Why concat `[s, t, s*t, s-t]` instead of just `s` and `t`?

A shallow MLP is bad at inventing cross-features on its own. If you hand it only the raw `s` (behavior) and `t` (target), it has to learn any notion of "similar" from scratch — and it only has `4·A → 2A → A → 1` parameters to do it with. So DIN pre-computes the two interactions that actually matter for relevance and hands them in as features:

- **`s ⊙ t`** — element-wise product, a coordinate-wise "do these agree?" signal. Large positive entries mean the behavior and the candidate share the same dimension of interest. This is the standard dot-product intuition, but kept per-dimension so the MLP can weight coordinates differently.
- **`s − t`** — the difference. Encodes how they differ, not just whether they match. Two behaviors could both be "close" to the target on average and still differ in important ways; the diff tells the MLP which axes are off.

The paper calls the concat "explicit knowledge to help relevance modeling" — that's really the whole story. Feature engineering the interaction terms lets the tiny activation MLP focus on combining them, not discovering them.

Reference:

[0] Zhou, G. et al. (2018). Deep Interest Network for Click-Through Rate Prediction, https://arxiv.org/abs/1706.06978
