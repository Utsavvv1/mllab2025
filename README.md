
---

# 📘 Improving Transformer Efficiency with Key/Value Projection
Machine Learning Lab – IIIT Pune (2025)

This project extends the original **“Attention Is All You Need”** Transformer architecture by introducing a more efficient variant of the self-attention mechanism. The modification is directly based on our proposal **“Improving Transformer Efficiency with Key/Value Projection”**, where we aim to reduce the **computational and memory overhead** of attention without altering the overall Transformer pipeline.

Reference proposal used:
> 📄 `/ML_Proposal (1).docx` — our internal project proposal outlining motivation, equations, expected results, and theoretical justification.

---

## 🚀 Objective

Self-attention in the original Transformer has **$O(n^2 \cdot d)$** complexity due to full-dimensional dot products between Query, Key, and Value matrices, where $n$ is the sequence length and $d$ is the model dimension. This becomes extremely expensive for long sequences.

Our goal was to:
* **Reduce computation**
* **Reduce memory footprint**
* Keep the model’s **external behavior unchanged**
* Achieve efficiency without modifying the encoder/decoder structure

We accomplish this by computing attention in a **lower-dimensional projected space**.

---

## 🧠 Proposed Improvement (from project proposal)

For each attention head, instead of computing attention using full-dimensional Key and Value vectors of size $d$, we project them into a smaller dimension **$k$**, where **$k \ll d$**.

* **Original Attention** 
* **Modified Attention** 

We modify the attention mechanism to:
1.  Project $K \rightarrow K'$ and $V \rightarrow V'$ using learned matrices $W_k$ and $W_v$.
2.  Project $Q \rightarrow Q'$ into the same reduced dimension $k$ using $W_q$.
3.  Compute attention using $Q'$ and $K'$.
4.  Use $V'$ to compute the compressed output $O'$.
5.  Project the result $O'$ back to the original dimension $d$ using $W_o$.

### Mathematical Formulation
The formulation for the Key/Value Projection (KVP) attention is:

$$
K' = KW_k,\quad V' = VW_v,\quad Q' = QW_q
$$

$$
\text{scores} = \frac{Q'{K'}^T}{\sqrt{k}}
$$

$$
O' = \text{softmax}(\text{scores}) \cdot V'
$$

$$
O = O'W_o
$$

This preserves the structure of the Transformer while reducing the inner attention complexity from $O(n^2 \cdot d)$ to **$O(n^2 \cdot k)$**. This formulation comes directly from the reference proposal document.

---

## 🔧 Code Changes (High-Level Overview)

Although the modification is isolated to a single module, implementing it required substantial internal restructuring of how attention is computed within the model.

All changes were made in: `transformer/Modules.py`

Specifically, the following engineering work was performed:
* Added a parallel **low-rank projection pathway** for $Q, K$, and $V$, each with new learned linear transformations to map into the reduced dimension $k$.
* Replaced the original full-dimensional attention computation with a compressed **bilinear attention operation**, requiring careful handling of tensor shapes across multi-head splits.
* Introduced an **output reconstruction projection** ($W_o$) to map the compressed attention result back into the original model dimension, ensuring seamless compatibility with the rest of the Transformer block.
* Refactored the internal forward-pass logic to synchronize reduced-dimension operations with masking, scaling, softmax, and residual connections.
* **Maintained full API compatibility** so no changes were needed in `Encoder`, `Decoder`, `Embeddings`, or training scripts — despite the attention mechanism being mathematically re-engineered.

In effect, the core of the Transformer’s attention engine was replaced with a more efficient, **low-rank variant**, while keeping external behavior identical.

---

## 📉 Expected Impact

* Lower compute cost for attention
* Lower GPU memory usage
* Faster training for long sequence lengths
* Negligible performance drop for reasonable $k$ values

---

## 📚 References Used

* Vaswani et al., *Attention Is All You Need* (2017)
* Our Project Proposal: *Improving Transformer Efficiency with Key/Value Projection* (internal document used as basis for equations and architecture changes)

---
