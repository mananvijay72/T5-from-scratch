from core.tensor import Tensor
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, hidden_dim):
        self.weight = Tensor(np.random.randn(vocab_size, hidden_dim), requires_grad=True)

    def __call__(self, input_ids):
        return Tensor(self.weight.data[input_ids.data], requires_grad=True)
    
    def parameters(self):
        return [self.weight]

class PositionalEmbedding:
    def __init__(self, max_seq_len, hidden_dim):
        self.weight = Tensor(np.random.randn(max_seq_len, hidden_dim), requires_grad=True)

    def __call__(self, seq_len):
        
        return Tensor(self.weight.data[:seq_len], requires_grad=True)
    
    def parameters(self):
        return [self.weight]