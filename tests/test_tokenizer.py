# scripts/test_tokenizer_print.py
from data.tokenizer import Tokenizer

def main():
    model_path = "spm.model"  # Adjust path if needed
    text = "Hello world. नमस्ते दुनिया।"

    tokenizer = Tokenizer(model_path)
    print(f"🔹 Original Text:\n{text}\n")

    token_ids = tokenizer.encode(text)
    print(f"🔸 Encoded Token IDs:\n{token_ids}\n")

    decoded_text = tokenizer.decode(token_ids)
    print(f"🔹 Decoded Text:\n{decoded_text}\n")

if __name__ == "__main__":
    main()