from core.tensor import Tensor
from model.encoder_block import EncoderBlock
import numpy as np

class Transformer:
    def __init__(self, config):
        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]

        self.embeddings = np.random.randn(self.vocab_size, self.d_model) * 0.01
        self.output_weights = np.random.randn(self.d_model, self.vocab_size) * 0.01

        self.encoder_blocks = [EncoderBlock(self.d_model, self.n_heads) for _ in range(self.n_layers)]

        # Learnable positional encoding
        self.max_seq_len = config.get("max_seq_len", 512)
        self.positional_encoding = Tensor(np.random.randn(self.max_seq_len, self.d_model), requires_grad=True)

    def forward(self, input_ids):
        x = np.array([self.embeddings[i] for i in input_ids])
        for block in self.encoder_blocks:
            x = block.forward(x)
        logits = x @ self.output_weights
        return logits

    def backward(self):
        pass  # Stub

    def step(self, lr):
        pass  # Stub