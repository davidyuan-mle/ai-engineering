# AI Engineering

The purpose is to understand and learn the foundations and applications of AI Engineering — building large language models from scratch and studying the core components that power modern generative AI.

## GPT-2 Implementation from Scratch

A full PyTorch implementation of the GPT-2 architecture, built component by component to deeply understand how large language models work under the hood.

### Architecture Components

- **Token & Positional Embeddings** — mapping vocabulary tokens and sequence positions into dense vector representations
- **Multi-Head Causal Self-Attention** — Q/K/V projections, scaled dot-product attention with causal masking, and output projection
- **Transformer Block** — pre-norm architecture with residual connections (LayerNorm → Attention → Residual → LayerNorm → FFN → Residual)
- **Feed-Forward Network (FFN)** — two-layer MLP with GELU activation and 4x expansion ratio
- **Layer Normalization** — custom implementation with learnable scale and shift parameters
- **GELU Activation** — Gaussian Error Linear Unit implemented from the mathematical formula
- **Text Generation** — autoregressive decoding with greedy sampling using tiktoken tokenizer

### Model Configuration (GPT-2 124M)

| Parameter | Value |
|---|---|
| Vocabulary size | 50,257 |
| Embedding dimension (d_model) | 768 |
| Context length | 1,024 |
| Attention heads | 12 |
| Transformer layers | 12 |

## Additional Topics

- Activation functions (ReLU, GELU, Swish, etc.)
- Deformable Convolutional Networks v2 (DCNv2)

## References

- Raschka, S. (2025). *Build a Large Language Model (From Scratch)*
- Karpathy, A. [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
