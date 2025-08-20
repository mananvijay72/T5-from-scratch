config = {
    "vocab_size": 15193,        # your existing vocab size
    "hidden_dim": 64,        # smaller than 256
    "encoder": {
        "num_layers": 2,      # fewer layers
        "hidden_dim": 64,
        "num_heads": 2,       # fewer attention heads
        "max_seq_len": 512,
        "ff_dim": 256,        # smaller feed-forward
        "vocab_size": 15193
    },
    "decoder": {
        "num_layers": 2,
        "hidden_dim": 64,
        "num_heads": 2,
        "max_seq_len": 512,
        "ff_dim": 256,
        "vocab_size": 15193
    },
}
