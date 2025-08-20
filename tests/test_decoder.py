import numpy as np
from encoder.t5encoder import T5Encoder
from decoder.t5decoder import T5Decoder
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

# Simulate encoder input
encoder_input = Tensor(np.random.randint(0, CONFIG["vocab_size"], (2, 8)))
encoder = T5Encoder(CONFIG)
encoder_out = encoder(encoder_input)  # shape: (2, 8, 12)

# Simulate decoder input: "<pad> <extra_id_0> fox <extra_id_1> dog"
decoder_input = Tensor(np.random.randint(0, CONFIG["vocab_size"], (2, 5)))  # shape: (2, 5)
decoder = T5Decoder(CONFIG)
logits = decoder(decoder_input, encoder_out)  # shape: (2, 5, vocab_size)

print("Decoder output shape:", logits.data.shape)

# Dummy target for loss
target = Tensor(np.random.randint(0, CONFIG["vocab_size"], (2, 5)))

# Simple cross-entropy loss
def cross_entropy(logits: Tensor, target: Tensor):
    # logits: (B, T, V)
    # target: (B, T)
    B, T, V = logits.data.shape

    # softmax
    shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)  # (B, T, V)

    # gather correct probs for each token
    batch_idx = np.arange(B)[:, None]        # shape (B, 1)
    time_idx = np.arange(T)[None, :]         # shape (1, T)
    chosen_probs = probs[batch_idx, time_idx, target.data]  # shape (B, T)

    # cross entropy
    log_probs = -np.log(chosen_probs + 1e-9)   # (B, T)
    loss_val = np.mean(log_probs)
    loss = Tensor(loss_val, requires_grad=True)

    # backward pass
    def _backward():
        if logits.requires_grad:
            grad = probs
            grad[batch_idx, time_idx, target.data] -= 1  # dL/dlogits
            grad /= (B * T)  # normalize over all tokens
            logits.grad = grad if logits.grad is None else logits.grad + grad

    loss._backward = _backward
    loss._prev = {logits}
    return loss


# Compute loss and backprop
loss = cross_entropy(logits, target)
print("Loss:", loss.data)
loss.backward()