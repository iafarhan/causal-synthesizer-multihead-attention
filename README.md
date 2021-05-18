## Causal and Synthesizer Multihead-Attention

We have implemented different variants of MultiheadAttention mechanisms. 
- CausalSelfAttention
- SynthesizerSelfAttention

**Causal Self-Attention** is the vanilla multi-head masked Self-attention layer with a projection at the end. we have used scaled-dot product as our scoring function in this case.

**Synthesizer Self-Attention** is a very recent alternative to causal self-attention that has potential benefits by removing this dot product. In vanilla self-attention the scoring function returns a block_size * block_size attention scores. This computation is quaratic in the sequence's length. Synthesizer self-attention overcomes this and computes the block_size * block_size matrix of attention scores directly. it is inspired from [Synthesizer: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)
