from data.tokenizer import Tokenizer

def main():
    input_file = "sample_data/multilingual.txt"
    tokenizer = Tokenizer(model_path=None)
    tokenizer.train(input_file, model_prefix="spm", vocab_size=42)
    print("✅ Tokenizer trained. Files generated: spm.model, spm.vocab")

if __name__ == "__main__":
    main()
    print("Tokenization training completed.")