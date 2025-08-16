from model.embeddings import EmbeddingLayer, PositionalEmbedding
from model.encoder_block import EncoderBlock
from core.tensor import Tensor
import numpy as np

class T5Encoder:
    def __init__(self, config):
        self.token_embed = EmbeddingLayer(config["vocab_size"], config["hidden_dim"])
        self.pos_embed = PositionalEmbedding(config["max_seq_len"], config["hidden_dim"])
        self.layers = [EncoderBlock(config) for _ in range(config["num_layers"])]

    def __call__(self, input_ids):  #input_id = batch x seq_len
        seq_len = input_ids.shape[1]
        x = self.token_embed(input_ids) + self.pos_embed(seq_len)
        for layer in self.layers:
            x = layer(x)
        return x