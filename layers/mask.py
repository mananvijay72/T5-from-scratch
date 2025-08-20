import cupy as np

def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask