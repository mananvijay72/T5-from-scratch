from core.tensor import Tensor
import cupy as np
from core.ops import relu

class FeedForward:
    def __init__(self, d_model, d_ff):
        # Xavier initialization
        limit1 = np.sqrt(6 / (d_model + d_ff))
        limit2 = np.sqrt(6 / (d_ff + d_model))
        
        self.w1 = Tensor(np.random.uniform(-limit1, limit1, (d_model, d_ff)), requires_grad=True)
        self.b1 = Tensor(np.zeros((1, d_ff)), requires_grad=True)

        self.w2 = Tensor(np.random.uniform(-limit2, limit2, (d_ff, d_model)), requires_grad=True)
        self.b2 = Tensor(np.zeros((1, d_model)), requires_grad=True)

    def __call__(self, x: Tensor):
        # (xW1 + b1) -> ReLU -> (xW2 + b2)
        return (relu(x @ self.w1 + self.b1)) @ self.w2 + self.b2
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]
