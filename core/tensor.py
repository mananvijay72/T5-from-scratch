# Autograd Tensor implementation for batch/matrix/3D data
# - Supports: +, -, neg, *, **, sqrt, @, /, T, reshape, exp, log, mean, gather, sum
# - sum and gather accept axis/dim arguments
# - Backprop handles broadcasting and batched matmul
# - Includes small tests demonstrating gradients on batch tensors
import numpy as np

# Utility helpers
def ensure_numpy(x):
    if isinstance(x, Tensor):
        return x.data
    elif isinstance(x, (np.ndarray, float, int)):
        return np.array(x) if not isinstance(x, np.ndarray) else x
    else:
        return np.array(x)

def _unbroadcast(grad, shape):
    # Sum out broadcasted dimensions so grad matches `shape`.
    if grad.shape == shape:
        return grad
    # Sum trailing dimensions introduced by broadcasting
    # Example: grad (2,3,4) -> shape (3,4) -> need to sum axis 0
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # For same ndim, sum axes where shape==1 but grad>1
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, Tensor):
            data = data.data.copy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data.astype(float)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for graph viz/debug
        self.shape = self.data.shape

    # Basic properties
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dim(self):
        return self.data.ndim

    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    return
                g = out.grad.T
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def reshape(self, *shape):
        new_shape = shape if len(shape) > 1 else shape[0] if isinstance(shape[0], (tuple, list)) else shape
        out = Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')
        old_shape = self.shape
        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    return
                g = out.grad.reshape(old_shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    # Representation
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # Operator helpers for binary ops with broadcasting
    def _binary_op(self, other, op_name, forward, grad_self, grad_other):
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        out_data = forward(self.data, other_data)
        requires = self.requires_grad or (other.requires_grad if isinstance(other, Tensor) else False)
        out = Tensor(
            out_data,
            requires_grad=requires,
            _children=tuple([t for t in (self, other) if isinstance(t, Tensor)]),
            _op=op_name
        )


        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = grad_self(out.grad, self.data, other_data)
                g_self = _unbroadcast(g_self, self.shape)
                self.grad = g_self if self.grad is None else self.grad + g_self
            if isinstance(other, Tensor) and other.requires_grad:
                g_other = grad_other(out.grad, self.data, other_data)
                g_other = _unbroadcast(g_other, other.shape)
                other.grad = g_other if other.grad is None else other.grad + g_other
        out._backward = _backward
        return out

    # Addition
    def __add__(self, other):
        return self._binary_op(other, 'add', lambda a,b: a + b,
                               lambda gout,a,b: gout,
                               lambda gout,a,b: gout)
    __radd__ = __add__

    # Subtraction
    def __sub__(self, other):
        return self._binary_op(other, 'sub', lambda a,b: a - b,
                               lambda gout,a,b: gout,
                               lambda gout,a,b: -gout)
    def __rsub__(self, other):
        # other - self
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return other_t.__sub__(self)

    # Negation
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op='neg')
        def _backward():
            if self.requires_grad and out.grad is not None:
                g = -out.grad
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    # Multiplication
    def __mul__(self, other):
        return self._binary_op(other, 'mul', lambda a,b: a * b,
                               lambda gout,a,b: gout * b,
                               lambda gout,a,b: gout * a)
    __rmul__ = __mul__

    # Division
    def __truediv__(self, other):
        return self._binary_op(other, 'div', lambda a,b: a / b,
                               lambda gout,a,b: gout / b,
                               lambda gout,a,b: -gout * a / (b**2))
    def __rtruediv__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return other_t.__truediv__(self)

    # Power
    def __pow__(self, power):
        if isinstance(power, Tensor):
            children = (self, power)
        else:
            children = (self,)
        power_val = power.data if isinstance(power, Tensor) else power
        out = Tensor(self.data ** power_val, requires_grad=self.requires_grad or (isinstance(power, Tensor) and power.requires_grad), _children=children, _op='pow')
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = out.grad * (power_val * (self.data ** (power_val - 1)))
                g_self = _unbroadcast(g_self, self.shape)
                self.grad = g_self if self.grad is None else self.grad + g_self
            if isinstance(power, Tensor) and power.requires_grad:
                # d/dp a^p = a^p * log(a)
                g_p = out.grad * (self.data ** power_val) * np.log(self.data + 1e-20)
                g_p = _unbroadcast(g_p, power.shape)
                power.grad = g_p if power.grad is None else power.grad + g_p
        out._backward = _backward
        return out

    # sqrt
    def sqrt(self):
        return self.__pow__(0.5)

    # exp, log
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')
        def _backward():
            if self.requires_grad and out.grad is not None:
                g = out.grad * out.data  # exp(x) derivative is exp(x)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-20), requires_grad=self.requires_grad, _children=(self,), _op='log')
        def _backward():
            if self.requires_grad and out.grad is not None:
                g = out.grad / (self.data + 1e-20)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    # Matrix multiplication (supports batched matmul via np.matmul)
    def __matmul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        out_data = self.data @ other_data
        requires = self.requires_grad or (other.requires_grad if isinstance(other, Tensor) else False)
        out = Tensor(
                out_data,
                requires_grad=requires,
                _children=tuple([t for t in (self, other) if isinstance(t, Tensor)]),
                _op='matmul'
            )

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                # grad wrt a: g @ b.T (with proper broadcasting)
                grad_a = g @ np.swapaxes(other_data, -1, -2)
                grad_a = _unbroadcast(grad_a, self.shape)
                self.grad = grad_a if self.grad is None else self.grad + grad_a
            if isinstance(other, Tensor) and other.requires_grad:
                grad_b = np.swapaxes(self.data, -1, -2) @ g
                grad_b = _unbroadcast(grad_b, other.shape)
                other.grad = grad_b if other.grad is None else other.grad + grad_b
        out._backward = _backward
        return out

    # Sum with axis/dim support
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='sum')
        def _backward():
            if self.requires_grad and out.grad is not None:
                grad = out.grad
                if not keepdims and axis is not None:
                    # need to reshape grad to have singleton dims where sum occurred
                    if isinstance(axis, int):
                        axes = (axis,)
                    else:
                        axes = tuple(axis)
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    grad = grad.reshape(shape)
                # broadcast to self.shape
                grad = np.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        out._backward = _backward
        return out

    # mean over axis(s)
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            if isinstance(axis, int):
                axes = (axis,)
            else:
                axes = tuple(axis)
            denom = 1
            for ax in axes:
                denom *= self.shape[ax]
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='mean')
        def _backward():
            if self.requires_grad and out.grad is not None:
                grad = out.grad
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        axes = (axis,)
                    else:
                        axes = tuple(axis)
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    grad = grad.reshape(shape)
                grad = np.broadcast_to(grad, self.shape) / denom
                self.grad = grad if self.grad is None else self.grad + grad
        out._backward = _backward
        return out

    # gather with dim/axis
    # index is expected to be integer Tensor with indices along `dim`
    def gather(self, dim, index):
        idx = index.data.astype(int) if isinstance(index, Tensor) else np.array(index).astype(int)

        # Forward: pick elements
        out_data = np.take_along_axis(
            self.data,
            np.expand_dims(idx, axis=-1) if self.data.ndim > idx.ndim else idx,
            axis=dim
        )

        out = Tensor(out_data,
                    requires_grad=self.requires_grad,
                    _children=(self,),
                    _op="gather")

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)

                # Scatter-add gradients back to the source tensor
                np.add.at(grad, tuple(idx.T) if idx.ndim > 1 else idx, out.grad)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out







    # Sum alias
    def sum_all(self):
        return self.sum(axis=None)

    # backward engine
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    if not isinstance(child, Tensor):
                        raise TypeError(f"Graph child is not a Tensor, got {type(child)}")
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

