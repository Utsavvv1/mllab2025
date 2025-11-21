''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module with Key-Value Compression.

    This implementation compresses the Key (K) and Value (V) projections from the
    original model dimension (d_model) to a smaller compressed dimension (k).
    
    Complexity Improvement:
    - Original: O(N^2 * d_model) for attention computation.
    - Compressed: O(N^2 * k) where k << d_model (e.g., 128 vs 512).
    
    This results in reduced runtime and memory usage while maintaining the 
    external interface (input/output dimension remains d_model).
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        
        # Compression logic:
        # We enforce a compressed dimension 'k' for internal attention computation.
        # This reduces the complexity of the attention map calculation (Q @ K.T)
        # and the weighted sum (Attn @ V).
        d_compressed = 128 
        
        # Recalculate per-head dimensions based on the compressed size.
        # d_k and d_v are now derived from d_compressed, not the input arguments.
        self.d_k = d_compressed // n_head
        self.d_v = d_compressed // n_head
        
        d_k = self.d_k
        d_v = self.d_v

        # Linear projections:
        # w_qs: Projects Q to compressed dimension (n_head * d_k)
        # w_ks: Projects K to compressed dimension (n_head * d_k) -> Reduces memory/compute
        # w_vs: Projects V to compressed dimension (n_head * d_v) -> Reduces memory/compute
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        # Output projection:
        # Projects from compressed dimension back to original d_model.
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # Note: The inner dimension here is the COMPRESSED dimension (k=128).
        # We reshape the output to (Batch, SeqLen, n_head, d_k) to separate heads.
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # We swap dimensions 1 and 2 to get (Batch, n_head, SeqLen, d_k).
        # This allows us to perform batch matrix multiplication across the head dimension.
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # Broadcast mask to account for the head dimension.
            # Mask shape: (Batch, 1, SeqLen) -> (Batch, 1, 1, SeqLen) or similar depending on input.
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Compute Attention with compressed vectors.
        # Complexity: O(N^2 * k) instead of O(N^2 * d_model)
        # Output q is the weighted sum of values: (Batch, n_head, SeqLen_Q, d_v)
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # We swap back to (Batch, SeqLen, n_head, d_v).
        # contiguous() is needed because transpose makes the tensor non-contiguous in memory.
        # view() combines n_head and d_v back into a single dimension: (Batch, SeqLen, n_head * d_v).
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        # Project back to original d_model dimension
        # The final linear layer mixes information from all heads.
        q = self.dropout(self.fc(q))
        
        # Residual Connection: Add the original input to the output.
        # This helps with gradient flow in deep networks.
        q += residual

        # Layer Normalization: Normalize the sum to have mean 0 and variance 1.
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
