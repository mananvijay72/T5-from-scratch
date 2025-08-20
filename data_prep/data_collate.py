# src/data_collate.py
import cupy as np

import cupy as np  # np is CuPy

def pad_seqs(seqs, pad_id=0):
    batch_size = len(seqs)
    max_len = max(len(s) for s in seqs)

    # pre-allocate CuPy array on GPU
    padded = np.full((batch_size, max_len), pad_id, dtype=np.int32)

    # copy each sequence into the array
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = np.array(s, dtype=np.int32)

    return padded



def make_batch_from_jsonl_lines(jsonl_lines, PAD_ID=0, EOS_ID=2, BOS_ID=1):
    EOS_ID = 2  # End of Sentence token
    BOS_ID = 1  # Beginning of Sentence token
    PAD_ID = 0  # Padding token

    # jsonl_lines: list of dicts with "input_ids" and "target_ids"
    enc_inputs = [np.array(x["input_ids"], dtype=np.int32) for x in jsonl_lines]
    tgt_ids    = [np.array(x["target_ids"], dtype=np.int32) for x in jsonl_lines]

    # decoder input: BOS + target_ids
    dec_inputs  = [np.concatenate([np.array([BOS_ID], dtype=np.int32), t]) for t in tgt_ids]
    # decoder output (labels): target_ids + EOS
    dec_outputs = [np.concatenate([t, np.array([EOS_ID], dtype=np.int32)]) for t in tgt_ids]

    # Pad all to uniform lengths
    enc_batch    = pad_seqs(enc_inputs, pad_id=PAD_ID)
    dec_in_batch = pad_seqs(dec_inputs, pad_id=PAD_ID)
    dec_out_batch= pad_seqs(dec_outputs, pad_id=PAD_ID)

    # Masks as np.ndarray (int32, but could be bool)
    enc_mask = (enc_batch != PAD_ID).astype(np.int32)
    dec_mask = (dec_in_batch != PAD_ID).astype(np.int32)

    return {
        "encoder_input": np.array(enc_batch, dtype=np.int32),   # shape (B, S_enc)
        "encoder_mask":  np.array(enc_mask, dtype=np.int32),
        "decoder_input": np.array(dec_in_batch, dtype=np.int32),# shape (B, S_dec)
        "decoder_output":np.array(dec_out_batch, dtype=np.int32),
        "decoder_mask":  np.array(dec_mask, dtype=np.int32)
    }

# Example usage:
if __name__ == "__main__":
    import json
    lines = []
    with open("data/processed/tokenized/train.jsonl", encoding="utf-8") as f:
        for _ in range(2):  # small batch of 2
            line = json.loads(f.readline())
            lines.append(line)
    batch = make_batch_from_jsonl_lines(lines)
    for k, v in batch.items():
        print(k, v.shape, type(v))
