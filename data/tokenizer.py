# tokenizer.py
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def encode(self, text, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def train(self, input_file, model_prefix="spm", vocab_size=32000):
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="unigram",
            bos_id=1,
            eos_id=2,
            pad_id=0,
            unk_id=3
        )
