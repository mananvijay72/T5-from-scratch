from core.tensor import Tensor
import numpy as np

class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads   # per-head dimension

        # Learnable weights
        self.W_q = Tensor(np.random.randn(d_model, d_model), requires_grad=True)
        self.W_k = Tensor(np.random.randn(d_model, d_model), requires_grad=True)
        self.W_v = Tensor(np.random.randn(d_model, d_model), requires_grad=True)
        self.W_o = Tensor(np.random.randn(d_model, d_model), requires_grad=True)

    def split_heads(self, x: Tensor, B, T):
        """
        x: [B, T, D]
        return: [B, n_heads, T, d_head]
        """
        reshaped = x.data.reshape(B, T, self.n_heads, self.d_head) #so that it reduce the last dimension to the front and then we transpose 
        #each head will work on lower embedding dimension as pe
        transposed = np.transpose(reshaped, (0, 2, 1, 3))  # [B, h, T, d_head]
        return Tensor(transposed, requires_grad=x.requires_grad)

    def combine_heads(self, x: Tensor, B, T):
        """
        x: [B, n_heads, T, d_head]
        return: [B, T, D]
        """
        transposed = np.transpose(x.data, (0, 2, 1, 3))  # [B, T, h, d_head]
        merged = transposed.reshape(B, T, self.d_model)  # [B, T, D]
        return Tensor(merged, requires_grad=x.requires_grad)

    def __call__(self, x: Tensor):
        B, T, D = x.data.shape
        assert D == self.d_model, "Input dim must match d_model"

        # Linear projections
        Q = x @ self.W_q  # [B, T, D]
        K = x @ self.W_k
        V = x @ self.W_v

        # Split into heads
        Q = self.split_heads(Q, B, T)  # [B, h, T, d_head]
        K = self.split_heads(K, B, T)
        V = self.split_heads(V, B, T)

        # Attention scores: QK^T / sqrt(d_head)
        K_T = np.transpose(K.data, (0, 1, 3, 2))  # [B, h, d_head, T]
        scores = (Q.data @ K_T) / np.sqrt(self.d_head)  # [B, h, T, T]

        # Softmax along last dim
        exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp / np.sum(exp, axis=-1, keepdims=True)  # [B, h, T, T]
        weights = Tensor(weights, requires_grad=True)

        # Weighted sum
        context = weights.data @ V.data   # [B, h, T, d_head]
        context = Tensor(context, requires_grad=True)

        # Merge heads
        context = self.combine_heads(context, B, T)  # [B, T, D]

        # Final output projection
        out = context @ self.W_o  # [B, T, D]
        return out

    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
