import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def train(self, input_file, model_prefix="spm", vocab_size=32000):
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="unigram"
        )