from core.tensor import Tensor
import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = Tensor(np.random.randn(d_model, d_ff), requires_grad=True)
        self.w2 = Tensor(np.random.randn(d_ff, d_model), requires_grad=True)

    def __call__(self, x: Tensor):
        return (x @ self.w1).relu() @ self.w2
    
    def parameters(self):
        return [self.w1, self.w2]