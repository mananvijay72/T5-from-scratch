import numpy as np
import sentencepiece as spm
from transformer.transformer import Transformer
from core.tensor import Tensor

# ---------------------------
# Load trained model
# ---------------------------
def load_model(config, weights_path="transformer_model.npz"):
    model = Transformer(config)
    weights = np.load(weights_path, allow_pickle=True)
    for idx, p in enumerate(model.parameters()):
        p.data = weights[f"param_{idx}"]
    print("✅ Model weights loaded from", weights_path)
    return model

# ---------------------------
# Greedy decoding for prediction
# ---------------------------
def predict(model, sp, text, sos_token=1, eos_token=2, max_len=50):
    """
    text: str (raw input sentence)
    sp: loaded SentencePiece model
    Returns: str (decoded output sentence)
    """

    # Encode input sentence to IDs
    input_ids = sp.encode(text, out_type=int)

    # Encoder input
    src = Tensor(np.array([input_ids]), requires_grad=False)

    # Start with <sos>
    decoder_input = [sos_token]

    for _ in range(max_len):
        dec = Tensor(np.array([decoder_input]), requires_grad=False)

        # Forward pass
        logits = model.forward(src, dec)

        # Get last token prediction
        next_token = int(np.argmax(logits.data[0, -1]))

        # Append token
        decoder_input.append(next_token)

        # Stop at <eos>
        if next_token == eos_token:
            break

    # Remove sos and eos before decoding
    output_ids = [t for t in decoder_input[1:] if t not in (sos_token, eos_token)]

    # Decode back to text
    return sp.decode(output_ids)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    config = {
        "vocab_size": 166,
        "hidden_dim": 256,
        "encoder": {"num_layers": 2, "hidden_dim": 256, "num_heads": 4, "max_seq_len": 512, "ff_dim": 512, "vocab_size": 166},
        "decoder": {"num_layers": 2, "hidden_dim": 256, "num_heads": 4, "max_seq_len": 512, "ff_dim": 512, 'vocab_size': 166},
    }

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=r"data/spm_multitask.model")

    # Load model
    model = load_model(config, "transformer_model.npz")

    # Example input
    input_text = "translate English to German: Hello, how are you?"
    output_text = predict(model, sp, input_text)

    print("Input:", input_text)
    print("Output:", output_text)
