import numpy as np
from core.tensor import Tensor


class EmbeddingLayer:
    """
    Standard embedding lookup: maps input token IDs -> hidden vectors.
    weight: (vocab_size, hidden_dim)
    input_ids: (batch, seq_len)
    output: (batch, seq_len, hidden_dim)
    """
    def __init__(self, vocab_size: int, hidden_dim: int):
        self.weight = Tensor(
            np.random.randn(vocab_size, hidden_dim) * 0.01,  # small init
            requires_grad=True
        )

    def __call__(self, input_ids):
        # Support both raw numpy arrays and Tensor-wrapped ids
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.data

        flat_ids = input_ids.reshape(-1)                      # (B*S,)
        gathered = self.weight.gather(dim=0, index=flat_ids)  # (B*S, H)

        out = gathered.reshape((
            input_ids.shape[0], input_ids.shape[1], self.weight.data.shape[1]
        ))  # (B, S, H)
        return out

    def parameters(self):
        return [self.weight]


class PositionalEmbedding:
    """
    Learnable positional embeddings.
    weight: (max_seq_len, hidden_dim)
    __call__(seq_len) -> (seq_len, hidden_dim)
    """
    def __init__(self, max_seq_len: int, hidden_dim: int):
        self.weight = Tensor(
            np.random.randn(max_seq_len, hidden_dim) * 0.01,
            requires_grad=True
        )

    def __call__(self, seq_len: int) -> Tensor:
        indices = np.arange(seq_len)  # shape (seq_len,)
        return self.weight.gather(dim=0, index=indices)

    def parameters(self):
        return [self.weight]
