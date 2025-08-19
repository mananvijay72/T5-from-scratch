from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward
from layers.layernorm import LayerNorm
from core.tensor import Tensor
import numpy as np

class DecoderBlock:
    def __init__(self, config):
        d_model = config["hidden_dim"]
        num_heads = config["num_heads"]
        ff_dim = config["ff_dim"]

        # Self-attention (masked)
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Cross-attention (queries from decoder, keys/values from encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # Feed Forward
        self.ff = FeedForward(d_model, ff_dim)

        # LayerNorms
        self.norm1 = LayerNorm(d_model)   # after masked self-attn
        self.norm2 = LayerNorm(d_model)   # after cross-attn
        self.norm3 = LayerNorm(d_model)   # after feedforward

    def __call__(self, x: Tensor, enc_out: Tensor, casual_mask = False, enc_padding_mask = None, dec_padding_mask = None):
        """
        x: [B, T, D] (decoder input embeddings)
        enc_out: [B, S, D] (encoder outputs, where S = src sequence length)
        """

        # 1. Masked Self-Attention (decoder attends to itself, but no peeking ahead)
        x = self.norm1(x + self.self_attn(x, casual_mask=True, padding_mask=dec_padding_mask))

        # 2. Cross-Attention (decoder attends to encoder outputs)
        x = self.norm2(x + self.cross_attn(x, context=enc_out, casual_mask=casual_mask, padding_mask=enc_padding_mask))

        # 3. Feed Forward
        x = self.norm3(x + self.ff(x))

        return x

    def parameters(self):
        return (
            self.self_attn.parameters() +
            self.cross_attn.parameters() +
            self.ff.parameters() +
            self.norm1.parameters() +
            self.norm2.parameters() +
            self.norm3.parameters()
        )
