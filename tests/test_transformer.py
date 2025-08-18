import numpy as np
from core.tensor import Tensor
from transformer.transformer import Transformer

def test_transformer_forward():
    # Small config for testing
    config = {
        "vocab_size": 20,
        "hidden_dim": 16,
        "max_seq_len": 10,
        "encoder": {
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 2,
            "ff_dim": 32,
            "max_seq_len": 10
        },
        "decoder": {
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 2,
            "ff_dim": 32,
            "max_seq_len": 10,
            "vocab_size": 20
        }
    }

    # Instantiate transformer
    model = Transformer(config)

    # Dummy input: batch=2, seq_len=5
    src_ids = Tensor(np.random.randint(0, config["vocab_size"], (2, 5)), requires_grad=False)
    tgt_ids = Tensor(np.random.randint(0, config["vocab_size"], (2, 5)), requires_grad=False)

    # Forward pass
    logits = model.forward(src_ids, tgt_ids)

    # Check shape: (batch, tgt_seq_len, vocab_size)
    assert logits.data.shape == (2, 5, config["vocab_size"]), f"Unexpected logits shape: {logits.data.shape}"

    # Check parameter collection
    params = model.parameters()
    assert len(params) > 0, "No parameters collected"

    def num_parameters(parameters):
        flat_params = []
        for p in parameters:
            if isinstance(p, list):
                flat_params.extend(p)
            else:
                flat_params.append(p)
        return sum(np.prod(p.data.shape) for p in flat_params)


    print("✅ Forward pass successful")
    print("Logits shape:", logits.data.shape)
    print("Number of parameters:", num_parameters(params))

if __name__ == "__main__":
    test_transformer_forward()
