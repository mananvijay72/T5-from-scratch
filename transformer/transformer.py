from encoder.t5encoder import T5Encoder
from decoder.t5decoder import T5Decoder
from layers.embeddings import EmbeddingLayer
from core.tensor import Tensor
import cupy as np
import numpy
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
        Works with both NumPy and CuPy tensors.
        """
        total = 0
        for p in self.parameters():
            if hasattr(p, "data"):   # our Tensor wrapper
                arr = p.data
            else:
                arr = p

            # Convert shape tuple to array before prod
            if isinstance(arr, np.ndarray):
                total += int(np.prod(np.array(arr.shape)))
            else:
                total += int(np.prod(arr.shape))

        return total

    


    def train(self, train_data_file, optimizer, learning_rate, loss_func, epochs=20, batch_size=50, save_path="T5.npz"):

        config = self.config

        # Load dataset on CPU (keep as list of dicts)
        dataset = load_jsonl(train_data_file)
        dataset_size = len(dataset)
        print(f"Loaded {dataset_size} samples.")

        # Initialize model
        model = Transformer(config)
        optimizer = optimizer(self.parameters(), lr=learning_rate)

        print(self.count_parameters(), "trainable parameters")

        # Training
        for epoch in range(epochs):
            # Shuffle dataset indices on CPU
            indices = numpy.arange(dataset_size)
            numpy.random.shuffle(indices)

            total_loss = 0
            steps = 0

            # Mini-batches
            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_data = [dataset[j] for j in batch_idx]

                # Prepare batch (CPU lists)
                batch = make_batch_from_jsonl_lines(batch_data)

                # Move batch to GPU using CuPy
                encoder_input = np.array(batch["encoder_input"], dtype=np.int32)
                decoder_input = np.array(batch["decoder_input"], dtype=np.int32)
                decoder_output = np.array(batch["decoder_output"], dtype=np.int32)
                encoder_mask = np.array(batch["encoder_mask"], dtype=np.int32)
                decoder_mask = np.array(batch["decoder_mask"], dtype=np.int32)

                # Convert to Tensors
                src_ids = Tensor(encoder_input, requires_grad=False)
                tgt_in = Tensor(decoder_input, requires_grad=False)
                tgt_out = Tensor(decoder_output)

                # Forward pass
                logits = self.forward(src_ids, tgt_in,
                                    enc_padding=encoder_mask,
                                    dec_padding=decoder_mask)

                # Flatten for cross-entropy
                B, S, V = logits.data.shape
                logits_flat = logits.reshape((B * S, V))
                targets_flat = tgt_out.reshape(-1)

                # Mask PAD tokens
                mask = targets_flat.data != 0
                logits_masked = logits_flat.gather(dim=0, index=mask)
                targets_masked = targets_flat.gather(dim=0, index=mask)

                # Compute loss
                loss = loss_func(logits_masked, targets_masked)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.data
                steps += 1

                #if steps % 10 == 0:
                    #print(f"Epoch {epoch+1}, Step {steps}, Loss: {loss.data:.4f}")

            avg_loss = total_loss / steps
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

            del src_ids, tgt_in, tgt_out, logits, logits_flat, targets_flat, mask
            np._default_memory_pool.free_all_blocks()
            np.get_default_memory_pool().free_all_blocks()


        # Save model weights on CPU
        weights = {}
        for idx, p in enumerate(self.parameters()):
            # Ensure weights are on CPU before saving
            data_cpu = p.data.get() if isinstance(p.data, np.ndarray) else p.data
            weights[f"param_{idx}"] = data_cpu
        numpy.savez(save_path, **weights)
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

