import numpy as np
from core.tensor import Tensor

def relu(x: Tensor) -> Tensor:
    """
    ReLU activation: max(0, x)
    """
    out_data = np.maximum(0, x.data)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="relu"
    )

    def _backward():
        if x.requires_grad:
            grad_mask = (x.data > 0).astype(x.data.dtype)
            x.grad = (x.grad + out.grad * grad_mask) if x.grad is not None else out.grad * grad_mask

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation: exp(x) / sum(exp(x), axis)
    Works for batched inputs too.
    """
    # Numerical stability: shift by max
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    out_data = exps / np.sum(exps, axis=axis, keepdims=True)

    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="softmax"
    )

    def _backward():
        if x.requires_grad:
            # Softmax Jacobian-vector product
            grad = out.grad
            dot = np.sum(out_data * grad, axis=axis, keepdims=True)
            dx = out_data * (grad - dot)
            x.grad = (x.grad + dx) if x.grad is not None else dx

    out._backward = _backward
    return out
