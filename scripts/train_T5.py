from linguaT5.train.trainer import Trainer

config = {
    "data_path": "sample_data/multilingual.txt",
    "max_spans": 1,
    "span_len": 2,
    "pad_id": 0,
    "mask_id": 999,
    "lr": 0.001,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "vocab_size": 32000
}

trainer = Trainer(config, tokenizer_model="spm.model")
trainer.train(steps=5)