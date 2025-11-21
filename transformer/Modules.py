import torch
import torch.nn as nn
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # Compute the raw attention scores: (Batch, Head, SeqLen_Q, SeqLen_K)
        # Formula: Q * K^T / sqrt(d_k)
        # Note: k.transpose(2, 3) swaps the last two dimensions for matrix multiplication.
        # This calculates the similarity between every Query and every Key.
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # Apply masking (e.g., for padding or future tokens in decoder).
            # We set masked positions to a very large negative number (-1e9).
            # When softmax is applied, exp(-1e9) becomes effectively zero, 
            # ensuring these positions do not contribute to the attention.
            attn = attn.masked_fill(mask == 0, -1e9)

        # Apply softmax to get probability distribution over Key/Value pairs.
        # The result sums to 1 along the last dimension (SeqLen_K).
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Compute the weighted sum of Values: (Batch, Head, SeqLen_Q, d_v)
        # Formula: Attention_Weights * V
        # Each output vector is a convex combination of Value vectors, weighted by attention scores.
        output = torch.matmul(attn, v)

        return output, attn
