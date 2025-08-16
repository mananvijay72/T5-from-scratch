# linguaT5/train/trainer.py
import time
import numpy as np
from model.transformer import Transformer
from data.pretrain_dataset import PretrainDataset

class Trainer:
    def __init__(self, config, tokenizer_model):
        self.model = Transformer(config)
        self.dataset = PretrainDataset(
            file_path=config["data_path"],
            tokenizer_model=tokenizer_model,
            max_spans=config["max_spans"],
            span_len=config["span_len"],
            pad_id=config["pad_id"],
            mask_id=config["mask_id"]
        )
        self.lr = config["lr"]

    def compute_loss(self, input_ids, target_ids_list):
        total_loss = 0
        for target_ids in target_ids_list:
            logits = self.model.forward(input_ids)
            # Simple cross-entropy over target span positions
            for i, target_id in enumerate(target_ids):
                if target_id != self.dataset.pad_id:
                    pred = logits[i]
                    prob = softmax(pred)
                    loss = -np.log(prob[target_id] + 1e-9)
                    total_loss += loss
        return total_loss / len(target_ids_list)

    def train(self, steps=10):
        for step in range(steps):
            idx = step % len(self.dataset)
            sample = self.dataset[idx]
            input_ids = sample["input"]
            target_ids_list = sample["targets"]

            loss = self.compute_loss(input_ids, target_ids_list)

            # Dummy gradient step
            self.model.backward()
            self.model.step(self.lr)

            print(f"Step {step+1} | Loss: {loss:.4f}")

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)