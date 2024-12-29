# Causal and Synthesizer Multihead Attention

We have implemented different variants of Multihead Attention mechanisms:

### 1. Causal Self-Attention
Causal Self-Attention is the vanilla multi-head masked self-attention layer with a projection at the end. It employs the scaled dot-product as the scoring function:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where:
- `Q`, `K`, and `V` are the query, key, and value matrices.
- `d_k` is the dimensionality of the key vectors.

This mechanism computes a `block_size × block_size` attention matrix, which makes the computation quadratic in the sequence length.

---

### 2. Synthesizer Self-Attention
Synthesizer Self-Attention is a recent alternative to causal self-attention that removes the need for pairwise dot-product operations. Instead, it directly computes the `block_size × block_size` matrix of attention scores:

```math
A = W_2 \sigma(W_1X + b_1) + b_2
```

Where:
- `W_1`, `W_2` are learnable weight matrices.
- `b_1`, `b_2` are biases.
- `\sigma` is a non-linear activation function.

Synthesizer Self-Attention reduces the quadratic computational cost associated with the scaled dot-product operation and offers an efficient alternative for long sequences.

### References
- Synthesizer: [Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)
