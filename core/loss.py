from core.tensor import Tensor
import numpy as np

def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    logits: Tensor of shape (N, C)
    targets: numpy array of shape (N,) with class indices
    """
    # Shift for stability
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    
    targets.data = targets.data.astype(np.int64)  # convert to int64 for indexing
    # Loss value
    N = logits.data.shape[0]
    loss_val = -np.log(probs[np.arange(N), targets.data] + 1e-9).mean()

    out = Tensor(loss_val, requires_grad=True, _children=(logits,), _op="cross_entropy")

    def _backward():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(N), targets.data] -= 1   # subtract 1 at correct class
            grad = grad / N                    # normalize
            logits.grad = (logits.grad + grad) if logits.grad is not None else grad

    out._backward = _backward
    return out
