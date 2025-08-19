# scripts/tokenize_and_split.py
import json
import random
from pathlib import Path
import sentencepiece as spm

IN = Path("data/processed/all_tagged.jsonl")
OUT_DIR = Path("data/processed/tokenized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sp = spm.SentencePieceProcessor()
sp.load("data/spm_multitask.model")

BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

# read
items = []
with IN.open(encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        items.append(obj)

random.shuffle(items)

# split 80/10/10 (tiny dataset - deterministic splits)
n = len(items)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_items = items[:n_train]
val_items = items[n_train:n_train + n_val]
test_items = items[n_train + n_val:]

def tokenize_obj(obj):
    inp = obj["input"]
    tgt = obj["target"]
    inp_ids = sp.encode(inp, out_type=int)
    tgt_ids = sp.encode(tgt, out_type=int)
    # add explicit BOS/EOS tokens as IDs around target for decoder
    # decoder input: [BOS] + tgt_ids  (model will shift internally or you will create decoder_input)
    # decoder output (labels): tgt_ids + [EOS]
    return {
        "input_ids": inp_ids,
        "target_ids": tgt_ids
    }

for split_name, split in [("train", train_items), ("val", val_items), ("test", test_items)]:
    outf = OUT_DIR / f"{split_name}.jsonl"
    with outf.open("w", encoding="utf-8") as fo:
        for obj in split:
            tok = tokenize_obj(obj)
            fo.write(json.dumps(tok, ensure_ascii=False) + "\n")
    print("Wrote", outf, "items:", len(split))
