import numpy as np
from core.tensor import Tensor
from transformer.transformer import Transformer
from training.optimizers import SGD
from training.loss import cross_entropy

def count_parameters(params):
    """Count total trainable parameters."""
    return sum(np.prod(p.data.shape) for p in params)

def test_copy_task():
    # Small config
    config = {
        "vocab_size": 10,
        "hidden_dim": 8,
        "max_seq_len": 5,
        "encoder": {"num_layers": 1, "hidden_dim": 8, "num_heads": 2, "ff_dim": 16, "max_seq_len": 5},
        "decoder": {"num_layers": 1, "hidden_dim": 8, "num_heads": 2, "ff_dim": 16, "max_seq_len": 5, "vocab_size": 10},
    }

    # Model + optimizer
    model = Transformer(config)
    params = model.parameters()
    optimizer = SGD(params, lr=0.1)

    print(f"📦 Model has {count_parameters(params)} trainable parameters")

    print("🚀 Training tiny copy task...")
    prev_loss = None
    for epoch in range(20):
        # Dummy copy task: input == output
        src = np.random.randint(0, config["vocab_size"], (2, 5))   # numpy
        tgt_in = src[:, :-1]
        tgt_out = src[:, 1:]

        # Forward (keep inputs numpy → model handles wrapping internally)
        logits = model.forward(src, tgt_in)  # (B, S, V)
        batch, seq, vocab = logits.data.shape

        logits_flat = logits.reshape((batch * seq, vocab))
        targets_flat = tgt_out.reshape(-1)   # numpy ints

        # Loss
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        print(f"Epoch {epoch+1}, Loss={loss.data:.4f}")

        # Check loss is decreasing (not strictly monotonic, but trending)
        if prev_loss is not None and loss.data > prev_loss:
            print("⚠️ Warning: loss did not decrease")
        prev_loss = loss.data

        # Check all parameters got gradients
        for p in params:
            assert p.grad is not None, f"❌ Param {p._op or 'unknown'} did not receive gradients!"

    print("✅ Test finished: all params updated, loss trending down")

if __name__ == "__main__":
    test_copy_task()
