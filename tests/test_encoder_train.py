import numpy as np
import time
from transformer.t5encoder import T5Encoder
from core.tensor import Tensor

CONFIG = {
    "vocab_size": 1000,
    "max_seq_len": 8,
    "hidden_dim": 12,
    "num_heads": 3,
    "ff_dim": 48,
    "num_layers": 2,
    "dropout": 0.0,
    "initializer_range": 0.02
}

# Dummy dataset: 10 samples, 3 classes
X_data = Tensor(np.random.randint(0, CONFIG["vocab_size"], (10, CONFIG["max_seq_len"]), dtype=np.int32))
Y_data = Tensor(np.random.randint(0, 3, size=(10,), dtype = np.int32))

# Model
encoder = T5Encoder(CONFIG)
W = Tensor(np.random.randn(CONFIG["hidden_dim"], 3) * 0.02, requires_grad=True)
def cross_entropy(logits: Tensor, target: Tensor):
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    log_probs = -np.log(probs[np.arange(len(target.data)), target.data])
    loss_val = np.mean(log_probs)
    loss = Tensor(loss_val, requires_grad=True)

    def _backward():
        if logits.requires_grad:
            grad = probs
            grad[np.arange(len(target.data)), target.data] -= 1
            grad /= len(target.data)
            logits.grad = grad if logits.grad is None else logits.grad + grad

    loss._backward = _backward
    loss._prev = {logits}
    return loss

# Training loop
lr = 0.1
for step in range(20):
    start = time.time()

    # Select batch
    idx = np.random.choice(X_data.shape[0], 2, replace=False) #batch size 2
    x = Tensor(X_data.data[idx], requires_grad=False)
    
    assert x.shape == (2, CONFIG["max_seq_len"])
    y = Tensor(Y_data.data[idx], requires_grad=False)

    # Forward
    encoded = encoder(x)  # (B, T, D)
    print(f"[Step {step}] Encoded shape: {encoded.data.shape}")

    pooled = Tensor(encoded.data.mean(axis=1))  # (B, D)

    logits = pooled @ W            # (B, 3)
    loss = cross_entropy(logits, y)

    # Backward
    loss.backward()

    # Update weights
    W.data -= lr * W.grad
    W.grad = None  # Reset

    # Logging
    duration = time.time() - start
    print(f"[Step {step}] Loss: {loss.data:.4f} | Time: {duration:.3f}s")

    # Convergence check
    if step > 0 and abs(prev_loss - loss.data) < 1e-4:
        print(f"[Step {step}] Converged.")
        break

    prev_loss = loss.data