from model.attention import MultiHeadSelfAttention
from model.feedforward import FeedForward
from model.layers import LayerNorm

class EncoderBlock:
    def __init__(self, config):
        self.attn = MultiHeadSelfAttention(config["hidden_dim"], config["num_heads"])
        self.ff = FeedForward(config["hidden_dim"], config["ff_dim"])
        self.norm1 = LayerNorm(config["hidden_dim"])
        self.norm2 = LayerNorm(config["hidden_dim"])
        

    def __call__(self, x):
        x = self.norm1(x + self.attn(x)) # add and norm after attention
        x = self.norm2(x + self.ff(x)) #add and norm after feedforward layer this new x will be input to next layer of encoder block  
        return x