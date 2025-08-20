import cupy as np

class SGD:
    def __init__(self, params, lr=0.01):
        # flatten nested lists
        self.params = self._flatten(params)
        self.lr = lr

    def _flatten(self, params):
        flat = []
        for p in params:
            if isinstance(p, (list, tuple)):
                flat.extend(self._flatten(p))
            else:
                flat.append(p)
        return flat

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
