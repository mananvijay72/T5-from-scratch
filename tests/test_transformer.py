from model.transformer import Transformer

config = {
    "d_model": 128,
    "vocab_size": 32000,
    "n_layers": 2,
    "n_heads": 4
}

model = Transformer(config)
input_ids = [10, 20, 30]
logits = model.forward(input_ids)
print("Logits shape:", logits.shape)  # Should be (3, vocab_size)