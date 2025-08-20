from datasets import load_dataset

# Load IITB English–Hindi dataset
ds = load_dataset("cfilt/iitb-english-hindi", split="train")

# Save to en_hi.txt
with open(r"C:\projects\T5\data\raw\en_hi.txt", "w", encoding="utf-8") as f:
    for sample in ds.select(range(1000)):
        english = sample["translation"]["en"].strip()
        hindi = sample["translation"]["hi"].strip()
        f.write(f"{english}\t{hindi}\n")

print("✅ Saved English–Hindi pairs to en_hi.txt")
