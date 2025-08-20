# scripts/train_spm.py
import sentencepiece as spm
from pathlib import Path
import json

RAW_JSONL = Path("data/processed/all_tagged.jsonl")
SPM_PREFIX = "data/spm_multitask"
VOCAB_SIZE = 946   # small for demo; use 32k+ for real runs

# 1. Build a plain text corpus for sentencepiece training by concatenating inputs+targets
corpus_txt = Path("data/processed/spm_corpus.txt")
with corpus_txt.open("w", encoding="utf-8") as fout, RAW_JSONL.open(encoding="utf-8") as fin:
    for line in fin:
        obj = json.loads(line)
        fout.write(obj["input"].strip() + "\n")
        fout.write(obj["target"].strip() + "\n")

# 2. Choose user-defined symbols (task and language tags) so they become single tokens
user_symbols = [
    "<translate>", "<summarize>", "<qa>",
    "<lang:en>", "<lang:hi>", "<lang:es>",
    "<pad>", "<bos>", "<eos>"
]
user_symbols_str = ",".join(user_symbols)

# 3. Train
spm.SentencePieceTrainer.Train(
    input=str(corpus_txt),
    model_prefix=str(SPM_PREFIX),
    vocab_size=VOCAB_SIZE,
    model_type="unigram",
    character_coverage=0.9995,
    user_defined_symbols=user_symbols_str,
    bos_id=1,
    eos_id=2,
    pad_id=0,
    unk_id=3
)

print("Trained:", SPM_PREFIX + ".model")
