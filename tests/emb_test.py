from core.tensor import Tensor
import numpy as np

# weight: (V, D), idx: (B, T) ints
V, D = 50, 8
B, T = 2, 3

weight = Tensor(np.random.randn(V, D), requires_grad=True)
idx = Tensor(np.random.randint(0, V, size=(B, T)))
emb = weight[idx]            # (B, T, D)
loss = emb.sum()             # simple scalar
loss.backward()

# Expect: weight.grad has accumulated ones at rows referenced by idx,
# of shape (V, D); rows not referenced should be zeros.
print("emb shape:", emb.data.shape)          # (2, 3, 8)
print("weight.grad shape:", weight.grad.shape)