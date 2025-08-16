from transformer.t5encoder import T5Encoder
import numpy as np

CONFIG = {
    "vocab_size": 32000,
    "max_seq_len": 8,
    "hidden_dim": 12,
    "num_heads": 8,
    "ff_dim": 2048,
    "num_layers": 6,
    "dropout": 0.1,
    "initializer_range": 0.02
}

def test_encoder():
    encoder = T5Encoder(CONFIG)
    dummy_input = np.random.randint(0, CONFIG["vocab_size"], (2, 8))
    print("Dummy input shape:", dummy_input.shape)
    output = encoder(dummy_input)
    print("Encoder output shape:", output.data.shape)

test_encoder()