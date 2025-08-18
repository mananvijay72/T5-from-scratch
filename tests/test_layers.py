import numpy as np
from core.tensor import Tensor
from encoder.embeddings import EmbeddingLayer, PositionalEmbedding

# ----- Test EmbeddingLayer -----
batch, seq_len, vocab_size, hidden_dim = 2, 3, 10, 4
embedding = EmbeddingLayer(vocab_size, hidden_dim)

# Fake input token IDs
input_ids = np.array([[1, 3, 5], [0, 2, 9]])
out = embedding(input_ids)

print("Embedding out shape:", out.data.shape)  # (2,3,4)

# Backward test
loss = out.sum()
loss.backward()
print("Grad shape on embedding weight:", embedding.weight.grad.shape)  # (10,4)

# ----- Test PositionalEmbedding -----
pos_emb = PositionalEmbedding(max_seq_len=6, hidden_dim=hidden_dim)
out2 = pos_emb(seq_len=3)
print("Positional embedding shape:", out2.data.shape)  # (3,4)

loss2 = out2.sum()
loss2.backward()
print("Grad shape on positional weight:", pos_emb.weight.grad.shape)  # (6,4)
