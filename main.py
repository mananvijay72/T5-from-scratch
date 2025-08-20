import sentencepiece as spm
from transformer.transformer import Transformer
import cupy as np
from core.loss import cross_entropy
from core.optimizers import SGD
from config import config
import os




spm = spm.SentencePieceProcessor(r"data/spm_multitask.model")


def load_model(weights_path="transformer_model.npz"):
    model = Transformer(config)
    weights = np.load(weights_path, allow_pickle=True)
    for idx, p in enumerate(model.parameters()):
        p.data = weights[f"param_{idx}"]
    print("✅ Model weights loaded from", weights_path)
    return model

def predict(text, model_path):

    model = load_model(weights_path= model_path)
    return model.predict(sp = spm, text= text, max_len=512)

def train_model(data_file, epochs=20, batch_size=20, lr=0.001, save_path="T5.npz"):

    model = Transformer(config)
    opt = SGD(model.parameters(), lr=lr)
    model.train(train_data_file = data_file, optimizer=SGD, loss_func=cross_entropy, epochs=epochs, batch_size=batch_size, learning_rate=lr, save_path=save_path)



if __name__ == "__main__":
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin")
    train_model(data_file="data/processed/tokenized/train.jsonl", epochs=50, batch_size = 20, save_path=r"T5_en_hi.npz")
    print(predict(text="Translate ENglish to hindi: I love you", model_path="T5_en_hi.npz"))
     