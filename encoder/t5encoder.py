from encoder.encoder_block import EncoderBlock
from layers.embeddings import PositionalEmbedding
from core.tensor import Tensor

class T5Encoder:
    def __init__(self, config):
        self.pos_embed = PositionalEmbedding(config["max_seq_len"], config["hidden_dim"])
        self.layers = [EncoderBlock(config) for _ in range(config["num_layers"])]

    def __call__(self, token_emb: Tensor, padding_mask = None):  
        seq_len = token_emb.data.shape[1]

        # Add positional embeddings
        x = token_emb + self.pos_embed(seq_len)

        # Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
