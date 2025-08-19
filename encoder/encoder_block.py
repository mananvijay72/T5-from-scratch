from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward
from layers.layernorm import LayerNorm

class EncoderBlock:
    def __init__(self, config):
        self.attn = MultiHeadAttention(config["hidden_dim"], config["num_heads"])
        self.ff = FeedForward(config["hidden_dim"], config["ff_dim"])
        self.norm1 = LayerNorm(config["hidden_dim"])
        self.norm2 = LayerNorm(config["hidden_dim"])
        

    def __call__(self, x, padding_mask=None):
        x = self.norm1(x + self.attn(x, padding_mask = padding_mask)) # add and norm after attention
        x = self.norm2(x + self.ff(x)) #add and norm after feedforward layer this new x will be input to next layer of encoder block  
        return x
    
    def parameters(self):
        params = []
        params += self.attn.parameters()
        params += self.ff.parameters()
        params += self.norm1.parameters()
        params += self.norm2.parameters()
        
        return params
