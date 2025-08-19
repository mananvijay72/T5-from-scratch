import json
import numpy as np
from transformer.transformer import Transformer
from data_prep.data_collate import make_batch_from_jsonl_lines
from core.tensor import Tensor
from core.optimizers import SGD  
from core.loss import cross_entropy
#from config import config

# ---------------------------
# Load dataset
# ---------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# ---------------------------
# Training Loop
# ---------------------------
def train(
    data_file,
    config,
    epochs=20,
    batch_size=2,
    lr=0.001,
    save_path="transformer_model.npz"
):

    # Load dataset
    dataset = load_jsonl(data_file)
    print(f"Loaded {len(dataset)} samples.")

    # Initialize model
    model = Transformer(config)
    optimizer = SGD(model.parameters(), lr=lr)

    print(model.count_parameters(), "trainable parameters")
    # Training
    for epoch in range(epochs):
        np.random.shuffle(dataset)

        total_loss = 0
        steps = 0

        # Mini-batches
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i : i + batch_size]
            batch = make_batch_from_jsonl_lines(batch_data)

            # Convert numpy arrays to Tensor
            src_ids = Tensor(batch["encoder_input"], requires_grad=False)
            tgt_in = Tensor(batch["decoder_input"], requires_grad=False)
            tgt_out = Tensor(batch["decoder_output"])  # labels are numpy

            # Forward pass
            logits = model.forward(src_ids, tgt_in,
                                   enc_padding=batch["encoder_mask"],
                                   dec_padding=batch["decoder_mask"])
            # logits: (B, S, V) → flatten for CE
            B, S, V = logits.data.shape
            logits_flat = logits.reshape((B * S, V))
            targets_flat = tgt_out.reshape(-1)


            # Mask out PAD tokens
            mask = targets_flat.data != 0

            logits_masked = logits_flat.gather(dim=0, index=mask)
            targets_masked = targets_flat.gather(dim=0, index=mask)

            # Loss
            loss = cross_entropy(logits_masked, targets_masked)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            steps += 1

            if steps % 10 == 0:
                print(f"Epoch {epoch+1}, Step {steps}, Loss: {loss.data:.4f}")

        avg_loss = total_loss / steps
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

    # Save model weights
    weights = {}
    for idx, p in enumerate(model.parameters()):
        weights[f"param_{idx}"] = p.data
    np.savez(save_path, **weights)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    config = {
        "vocab_size": 166,
        "hidden_dim": 256,
        "encoder": {"num_layers": 2, "hidden_dim": 256, "num_heads": 4, "max_seq_len": 512, "ff_dim": 512, "vocab_size": 166},
        "decoder": {"num_layers": 2, "hidden_dim": 256, "num_heads": 4, "max_seq_len": 512, "ff_dim": 512, 'vocab_size': 166},
    }

    train(r"data\processed\tokenized\train.jsonl", config, epochs=300, batch_size=2, lr=1e-3)
