import numpy as np

def pad_sequences(sequences, pad_id=0):
    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded