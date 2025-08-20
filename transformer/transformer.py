from encoder.t5encoder import T5Encoder
from decoder.t5decoder import T5Decoder
from layers.embeddings import EmbeddingLayer
from core.tensor import Tensor
import numpy as np
from utils.utils import load_jsonl
from data_prep.data_collate import make_batch_from_jsonl_lines


class Transformer:
    def __init__(self, config):
        self.token_embed = EmbeddingLayer(config["vocab_size"], config["hidden_dim"])
        self.encoder = T5Encoder(config["encoder"])
        self.decoder = T5Decoder(config["decoder"])
        self.config = config

    def forward(self, src_ids: Tensor, tgt_ids: Tensor, enc_padding = None, dec_padding = None):
        # Shared token embedding
        src_emb = self.token_embed(src_ids)
        tgt_emb = self.token_embed(tgt_ids)

        # Encoder forward pass
        encoder_out = self.encoder(src_emb, padding_mask= enc_padding)

        # Decoder forward pass (conditioned on encoder output)
        logits = self.decoder(tgt_emb, encoder_out, casual_mask = True, enc_padding_mask= enc_padding ,dec_padding_mask= dec_padding)
        return logits

    def parameters(self):
        params = []
        params += self.token_embed.parameters()
        params += self.encoder.parameters()
        params += self.decoder.parameters()
        return params
    
    def count_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        total = 0
        for p in self.parameters():
            if hasattr(p, "data"):   # our Tensor wrapper
                total += np.prod(p.data.shape)
            else:  # if it's raw numpy array
                total += np.prod(np.shape(p))
        return total
    
    def train(self, train_data_file, optimizer, learning_rate, loss_func, epochs = 20, batch_size = 50, save_path = "T5.npz"):

        config = self.config

        # Load dataset
        dataset = load_jsonl(train_data_file)
        print(f"Loaded {len(dataset)} samples.")

        # Initialize model
        model = Transformer(config)
        optimizer = optimizer(self.parameters(), lr=learning_rate)

        print(self.count_parameters(), "trainable parameters")
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
                logits = self.forward(src_ids, tgt_in,
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
                loss = loss_func(logits_masked, targets_masked)

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
        for idx, p in enumerate(self.parameters()):
            weights[f"param_{idx}"] = p.data
        np.savez(save_path, **weights)
        print(f"Model saved to {save_path}")

    def predict(self, sp, text, sos_token=1, eos_token=2, max_len=50):
        """
        text: str (raw input sentence)
        sp: loaded SentencePiece model
        Returns: str (decoded output sentence)
        """
        print("predicting.....")
        # Encode input sentence to IDs
        input_ids = sp.encode(text, out_type=int)

        # Encoder input
        src = Tensor(np.array([input_ids]), requires_grad=False)

        # Start with <sos>
        decoder_input = [sos_token]

        for _ in range(max_len):
            dec = Tensor(np.array([decoder_input]), requires_grad=False)

            # Forward pass
            logits = self.forward(src, dec)

            # Get last token prediction
            next_token = int(np.argmax(logits.data[0, -1]))

            # Append token
            decoder_input.append(next_token)

            # Stop at <eos>
            if next_token == eos_token:
                break

        # Remove sos and eos before decoding
        output_ids = [t for t in decoder_input[1:] if t not in (sos_token, eos_token)]
        print(output_ids)
        print("OUT PREDICT IN TRANSFORMER MODEL")
        # Decode back to text
        return sp.decode(output_ids)        

