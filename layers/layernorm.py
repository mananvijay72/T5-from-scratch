from core.tensor import Tensor
import cupy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, d_model)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, d_model)), requires_grad=True)

    def __call__(self, x: Tensor):
        mean = Tensor(np.mean(x.data, axis=-1, keepdims=True))
        var = Tensor(np.var(x.data, axis=-1, keepdims=True))
        norm = (x - mean) / (var + self.eps).sqrt()  # using exp for sqrt approximation
        return self.gamma * norm + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]