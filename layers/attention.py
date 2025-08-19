import numpy as np
from core.tensor import Tensor
from core.ops import softmax


class MultiHeadAttention:
    def __init__(self, hidden_dim, num_heads):
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Weight matrices as Tensors with gradients
        self.W_q = Tensor(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim), requires_grad=True)
        self.W_k = Tensor(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim), requires_grad=True)
        self.W_v = Tensor(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim), requires_grad=True)
        self.W_o = Tensor(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim), requires_grad=True)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split [B, T, D] into [B, num_heads, T, head_dim]
        """
        B, T, D = x.data.shape
        reshaped = x.data.reshape(B, T, self.num_heads, self.head_dim)
        transposed = reshaped.transpose(0, 2, 1, 3)  # [B, H, T, d]
        return Tensor(transposed, requires_grad=x.requires_grad)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combine [B, num_heads, T, head_dim] -> [B, T, D]
        """
        B, H, T, d = x.data.shape
        transposed = x.data.transpose(0, 2, 1, 3).reshape(B, T, H * d)
        return Tensor(transposed, requires_grad=x.requires_grad)

    def __call__(self, x: Tensor, context: Tensor = None, casual_mask: bool = False, padding_mask=None) -> Tensor:
        """
        x: [B, T, D] (queries always come from here)
        context: [B, S, D] (for cross-attention; if None, use x itself)
        mask: True for decoder self-attention (causal mask)
        padding_mask: [B, S] with 1 for real tokens, 0 for padding
        """
        if context is None:
            context = x  # self-attention

        # Linear projections
        Q = x @ self.W_q
        K = context @ self.W_k
        V = context @ self.W_v

        # Split heads
        Q = self._split_heads(Q)  # [B, H, T, d]
        K = self._split_heads(K)  # [B, H, S, d]
        V = self._split_heads(V)  # [B, H, S, d]

        # Attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) / Tensor(self.head_dim).sqrt()  # [B, H, T, S]

        # Apply causal mask (for decoder self-attention)
        if casual_mask:
            T, S = scores.shape[2], scores.shape[3]
            causal_mask = np.triu(np.ones((T, S)), k=1).astype(np.bool_)
            scores.data = np.where(causal_mask, -1e9, scores.data)

        # Apply padding mask (ignore <pad> tokens in context)
        if padding_mask is not None:
            # padding_mask: [B, S] -> expand to [B, 1, 1, S]
            expanded_mask = padding_mask[:, np.newaxis, np.newaxis, :]
            scores.data = np.where(expanded_mask == 0, -1e9, scores.data)

        # Softmax

        attn_weights = softmax(scores, axis=-1)

        '''attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)'''

        out = attn_weights @ V.data  # [B, H, T, d]
        out = Tensor(out, requires_grad=True)

        # Combine heads
        out = self._combine_heads(out)  # [B, T, D]

        # Final projection
        out = Tensor(out.data @ self.W_o.data, requires_grad=True)
        return out
    
    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
