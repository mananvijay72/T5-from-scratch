import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        visited, topo = set(), []

        def build(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build(child)
                topo.append(t)

        build(self)
        for t in reversed(topo):
            t._backward()

    @property
    def shape(self):
        return self.data.shape

    # ─── Operator Overloads ─────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __radd__(self, other): return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad - out.grad if other.grad is not None else -out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rsub__(self, other): return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = out.grad * self.data
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rmul__(self, other): return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = -out.grad * self.data / (other.data ** 2)
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rtruediv__(self, other): return Tensor(other) / self

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def sqrt(self):

        sqrt_data = np.sqrt(self.data)
        out = Tensor(sqrt_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = 0.5 / np.sqrt(self.data)
                self.grad = self.grad + grad * out.grad if self.grad is not None else grad * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    # ─── Activation Functions ───────────────────────────────────────────

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad * out.data
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out

    def softmax(self):
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Tensor(probs, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                s = out.data
                dx = np.empty_like(grad)
                for i in range(s.shape[0]):
                    si = s[i].reshape(-1, 1)
                    jac = np.diagflat(si) - si @ si.T
                    dx[i] = jac @ grad[i]
                self.grad = self.grad + dx if self.grad is not None else dx

        out._backward = _backward
        out._prev = {self}
        return out