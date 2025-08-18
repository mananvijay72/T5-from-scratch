from core.tensor import Tensor
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, d_model)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, d_model)), requires_grad=True)

    def __call__(self, x: Tensor):
        # Compute mean/var along last dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)

        # Normalize
        normed = (x.data - mean) / np.sqrt(var + self.eps)

        # Wrap back into Tensor
        normed = Tensor(normed, requires_grad=x.requires_grad)

        # Scale + Shift
        return self.gamma * normed + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]
