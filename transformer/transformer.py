from transformer.t5encoder import T5Encoder
from transformer.t5decoder import T5Decoder
from encoder.embeddings import EmbeddingLayer
from core.tensor import Tensor

class Transformer:
    def __init__(self, config):
        self.token_embed = EmbeddingLayer(config["vocab_size"], config["hidden_dim"])
        self.encoder = T5Encoder(config["encoder"])
        self.decoder = T5Decoder(config["decoder"])
        self.config = config

    def forward(self, src_ids: Tensor, tgt_ids: Tensor):
        # Shared token embedding
        src_emb = self.token_embed(src_ids)
        tgt_emb = self.token_embed(tgt_ids)

        # Encoder forward pass
        encoder_out = self.encoder(src_emb)

        # Decoder forward pass (conditioned on encoder output)
        logits = self.decoder(tgt_emb, encoder_out)
        return logits

    def parameters(self):
        params = []
        params += self.token_embed.parameters()
        params += self.encoder.parameters()
        params += self.decoder.parameters()
        return params
