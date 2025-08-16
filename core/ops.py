from core.tensor import Tensor
import numpy as np

def add(a: Tensor, b: Tensor):
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = a.grad + out.grad if a.grad is not None else out.grad
        if b.requires_grad:
            b.grad = b.grad + out.grad if b.grad is not None else out.grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def matmul(a: Tensor, b: Tensor):
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = a.grad + out.grad @ b.data.T if a.grad is not None else out.grad @ b.data.T
        if b.requires_grad:
            b.grad = b.grad + a.data.T @ out.grad if b.grad is not None else a.data.T @ out.grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def relu(a: Tensor):
    out = Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            grad = out.grad * (a.data > 0)
            a.grad = a.grad + grad if a.grad is not None else grad

    out._backward = _backward
    out._prev = {a}
    return out

def exp(a: Tensor):
    out = Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            grad = out.grad * out.data
            a.grad = a.grad + grad if a.grad is not None else grad

    out._backward = _backward
    out._prev = {a}
    return out

def softmax(t: Tensor):
    shifted = t.data - np.max(t.data, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    out = Tensor(probs, requires_grad=t.requires_grad)

    def _backward():
        if t.requires_grad:
            grad = out.grad
            s = out.data
            dx = np.empty_like(grad)
            for i in range(s.shape[0]):
                si = s[i].reshape(-1, 1)
                jac = np.diagflat(si) - si @ si.T
                dx[i] = jac @ grad[i]
            t.grad = t.grad + dx if t.grad is not None else dx

    out._backward = _backward
    out._prev = {t}
    return out

def sub(a: Tensor, b: Tensor):
    out = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = a.grad + out.grad if a.grad is not None else out.grad
        if b.requires_grad:
            b.grad = b.grad - out.grad if b.grad is not None else -out.grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def mul(a: Tensor, b: Tensor):
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            grad = out.grad * b.data
            a.grad = a.grad + grad if a.grad is not None else grad
        if b.requires_grad:
            grad = out.grad * a.data
            b.grad = b.grad + grad if b.grad is not None else grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def div(a: Tensor, b: Tensor):
    out = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            grad = out.grad / b.data
            a.grad = a.grad + grad if a.grad is not None else grad
        if b.requires_grad:
            grad = -out.grad * a.data / (b.data ** 2)
            b.grad = b.grad + grad if b.grad is not None else grad

    out._backward = _backward
    out._prev = {a, b}
    return out