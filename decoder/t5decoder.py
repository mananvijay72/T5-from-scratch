import numpy as np
from decoder.decoder_block import DecoderBlock
from layers.embeddings import PositionalEmbedding
from core.tensor import Tensor

class T5Decoder:
    def __init__(self, config):
        self.pos_embed = PositionalEmbedding(config["max_seq_len"], config["hidden_dim"])
        self.blocks = [DecoderBlock(config) for _ in range(config["num_layers"])]
        self.proj = Tensor(np.random.randn(config["hidden_dim"], config["vocab_size"]) * 0.02, requires_grad=True)

    def __call__(self, token_emb: Tensor, encoder_out: Tensor, casual_mask = True, enc_padding_mask = None, dec_padding_mask = None):
        seq_len = token_emb.data.shape[1]

        # Add positional embeddings
        x = token_emb + self.pos_embed(seq_len)

        # Pass through decoder blocks (with cross-attention on encoder output)
        for block in self.blocks:
            x = block(x, encoder_out, casual_mask = casual_mask, enc_padding_mask = enc_padding_mask, dec_padding_mask = dec_padding_mask)

        # Project to vocab logits
        logits = x @ self.proj  
        return logits

    def parameters(self):
        params = [self.proj]
        for block in self.blocks:
            params += block.parameters()
        return params
