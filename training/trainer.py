import numpy as np

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, src, tgt_in, tgt_out):
        # Forward
        logits = self.model.forward(src, tgt_in)

        # Compute loss
        loss = self.loss_fn(logits, tgt_out)

        # Backward
        loss.backward()

        # Update
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.data

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt_in, tgt_out in data:
                loss = self.train_step(src, tgt_in, tgt_out)
                total_loss += loss
            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
