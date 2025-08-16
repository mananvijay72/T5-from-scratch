# linguaT5/data/pretrain_dataset.py
import random
from data.tokenizer import Tokenizer
from data.loader import pad_sequences

class PretrainDataset:
    def __init__(self, file_path, tokenizer_model, max_spans=2, span_len=3, pad_id=0, mask_id=999):
        self.tokenizer = Tokenizer(tokenizer_model)
        self.max_spans = max_spans
        self.span_len = span_len
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.samples = self._load_text(file_path)

    def _load_text(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    def _corrupt_spans(self, tokens):
        corrupted = tokens.copy()
        targets = []
        if len(tokens) < self.span_len + 1:
            return corrupted, targets

        span_indices = random.sample(range(len(tokens) - self.span_len), k=min(self.max_spans, len(tokens) // self.span_len))
        for idx in span_indices:
            corrupted[idx:idx+self.span_len] = [self.mask_id]
            targets.append(tokens[idx:idx+self.span_len])
        return corrupted, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text)
        corrupted, targets = self._corrupt_spans(tokens)
        batch = pad_sequences([corrupted] + targets, pad_id=self.pad_id)
        return {
            "input": batch[0],
            "targets": batch[1:]
        }