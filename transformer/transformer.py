from encoder.t5encoder import T5Encoder
from decoder.t5decoder import T5Decoder
from layers.embeddings import EmbeddingLayer
from core.tensor import Tensor
import numpy as np

class Transformer:
    def __init__(self, config):
        self.token_embed = EmbeddingLayer(config["vocab_size"], config["hidden_dim"])
        self.encoder = T5Encoder(config["encoder"])
        self.decoder = T5Decoder(config["decoder"])
        self.config = config

    def forward(self, src_ids: Tensor, tgt_ids: Tensor, enc_padding = None, dec_padding = None):
        # Shared token embedding
        src_emb = self.token_embed(src_ids)
        tgt_emb = self.token_embed(tgt_ids)

        # Encoder forward pass
        encoder_out = self.encoder(src_emb, padding_mask= enc_padding)

        # Decoder forward pass (conditioned on encoder output)
        logits = self.decoder(tgt_emb, encoder_out, casual_mask = True, enc_padding_mask= enc_padding ,dec_padding_mask= dec_padding)
        return logits

    def parameters(self):
        params = []
        params += self.token_embed.parameters()
        params += self.encoder.parameters()
        params += self.decoder.parameters()
        return params
    
    def count_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        total = 0
        for p in self.parameters():
            if hasattr(p, "data"):   # our Tensor wrapper
                total += np.prod(p.data.shape)
            else:  # if it's raw numpy array
                total += np.prod(np.shape(p))
        return total
