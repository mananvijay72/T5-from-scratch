import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor(data={self.data})"
    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)

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

    # ─── Basic Ops ──────────────────────────
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

    def __sub__(self, other): return self + (-1 * other)
    def __rsub__(self, other): return Tensor(other) - self
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Tensor(other) / self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * other.data if self.grad is not None else out.grad * other.data
            if other.requires_grad:
                other.grad = other.grad + out.grad * self.data if other.grad is not None else out.grad * self.data
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad / other.data if self.grad is not None else out.grad / other.data
            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data ** 2)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev.update([self, other])

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.swapaxes(-1, -2)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

            if other.requires_grad:
                grad_other = self.data.swapaxes(-1, -2) @ out.grad
                other.grad = (other.grad + grad_other) if other.grad is not None else grad_other

        out._backward = _backward
        return out


    def sqrt(self):
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = (0.5 / np.sqrt(self.data)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        return out


    # ─── Activations ──────────────────────────
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
                self.grad = self.grad + out.grad * out.data if self.grad is not None else out.grad * out.data
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

    def transpose(self, *axes):
        """
        Returns a transposed tensor.
        If no axes are given, reverses the dimensions.
        """
        return Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self_grad = out.grad.reshape(self.data.shape)
                self.grad = (self.grad + self_grad) if self.grad is not None else self_grad

        out._backward = _backward
        out._prev = {self}
        return out


    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad / np.prod(self.shape)
                self.grad = np.ones_like(self.data) * grad

        out._backward = _backward
        out._prev = {self}
        return out




    def log(self):
        out = Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad / (self.data + 1e-9) if self.grad is not None else out.grad / (self.data + 1e-9)
        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad = out.grad
                self.grad = (self.grad + np.ones_like(self.data) * grad) if self.grad is not None else np.ones_like(self.data) * grad
        out._backward = _backward
        out._prev = {self}
        return out
    
    def _unbroadcast(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev.add(self)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad - out.grad) if self.grad is not None else -out.grad

        out._backward = _backward
        return out
    
    
    
    def _add_grad(self, g):
        """Accumulate gradient into this Tensor."""
        if g is None:
            return
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    def gather(self, dim, index):
        # unwrap index if it's a Tensor
        if isinstance(index, Tensor):
            index = index.data
        index = np.asarray(index)

        if dim == 0:
            # forward
            out_data = self.data[index]                 # shape: index.shape + trailing_shape
            out = Tensor(out_data, requires_grad=self.requires_grad)
            out._prev.add(self)

            def _backward():
                if not self.requires_grad:
                    return
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                flat_idx = index.reshape(-1)                # M = number of selected rows
                trailing_shape = self.data.shape[1:]       # e.g. (H,)
                prod_trailing = int(np.prod(trailing_shape)) if len(trailing_shape) > 0 else 1

                # make sure out.grad is a numpy array
                g = np.asarray(out.grad)

                # Flatten leading dims so we get (R, prod_trailing)
                flat_grad = g.reshape(-1, prod_trailing)  # R x prod_trailing

                # R must be a multiple of M. If R==M, 1-to-1. If R = M * G, then
                # we have G groups per index (e.g. earlier extra dims), sum them.
                R = flat_grad.shape[0]
                if R % flat_idx.size != 0:
                    # unable to align gradients with indices — produce helpful message
                    raise ValueError(
                        f"gather-backward: mismatch between flattened grad rows (R={R}) "
                        f"and number of indices (M={flat_idx.size})."
                    )

                groups = R // flat_idx.size
                if groups == 1:
                    grad_rows = flat_grad.reshape(flat_idx.size, *trailing_shape)
                else:
                    # reshape to (M, groups, prod_trailing) and sum groups -> (M, prod_trailing)
                    grad_rows = flat_grad.reshape(flat_idx.size, groups, prod_trailing).sum(axis=1)
                    grad_rows = grad_rows.reshape(flat_idx.size, *trailing_shape)

                # scatter-add back into embedding matrix rows
                np.add.at(self.grad, flat_idx, grad_rows)

            out._backward = _backward
            return out

        elif dim == 1:
            # forward: expect self.data shape (N, C) and index shape (N,) or (N,1)
            if self.data.ndim != 2:
                raise NotImplementedError("gather(dim=1) requires 2D tensor")
            index_arr = index
            if index_arr.ndim == 1:
                index_arr = index_arr[:, None]  # (N,1)
            out_data = np.take_along_axis(self.data, index_arr, axis=1)  # (N,1) or (N,k)
            out = Tensor(out_data, requires_grad=self.requires_grad)
            out._prev.add(self)

            def _backward():
                if not self.requires_grad:
                    return
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad_cols = np.asarray(out.grad)
                # collapse any extra dims beyond (N,k) by summing them
                if grad_cols.ndim > 2:
                    # sum all extra middle dims so shape becomes (N, k)
                    extra_axes = tuple(range(1, grad_cols.ndim - 1))
                    grad_cols = grad_cols.sum(axis=extra_axes, keepdims=False)

                if grad_cols.ndim == 1:
                    grad_cols = grad_cols[:, None]

                rows = np.arange(self.data.shape[0])[:, None]  # (N,1)
                np.add.at(self.grad, (rows, index_arr), grad_cols)

            out._backward = _backward
            return out

        else:
            raise NotImplementedError("Tensor.gather only supports dim=0 or dim=1.")








    

#----------------------------------------------------------------------------------------


def embedding_lookup(weight, indices):
    """
    weight: Tensor of shape (V, D...)
    indices: Tensor or np.ndarray/ints of shape (B, T, ...)
    returns: Tensor of shape (B, T, ..., D...)
    """
    # Normalize indices to a numpy integer array
    if isinstance(indices, Tensor):
        idx = indices.data
    else:
        idx = indices
    idx = np.asarray(idx)
    if idx.dtype.kind not in ("i", "u"):
        idx = idx.astype(np.int64)

    out_data = np.take(weight.data, idx, axis=0)
    out = Tensor(out_data, requires_grad=weight.requires_grad)

    if weight.requires_grad:
        V = weight.data.shape[0]
        trailing = int(np.prod(weight.data.shape[1:], dtype=np.int64))

        def _backward():
            if out.grad is None:
                return
            grad_in = np.zeros_like(weight.data)
            flat_idx = idx.reshape(-1)                          # (N,)
            flat_grad = out.grad.reshape(flat_idx.shape[0], -1) # (N, trailing)
            # Scatter-add; supports repeated indices
            np.add.at(grad_in.reshape(V, trailing), flat_idx, flat_grad)
            weight._add_grad(grad_in)

        out._backward = _backward
        out._prev = {weight}

    return out


def slice_rows(x: Tensor, stop: int, start: int = 0, step: int = 1):
    sl = slice(start, stop, step)
    out_data = x.data[sl]
    out = Tensor(out_data, requires_grad=x.requires_grad)

    if x.requires_grad:
        def _backward():
            if out.grad is None:
                return
            grad_in = np.zeros_like(x.data)
            # Collapse all leading dims except the slice axis
            np.add.at(grad_in, 
                      np.arange(start, stop, step), 
                      out.grad.reshape(-1, grad_in.shape[1]))
            x._add_grad(grad_in)

        out._backward = _backward
        out._prev = {x}

    return out

