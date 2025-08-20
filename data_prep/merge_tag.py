# scripts/merge_and_tag.py
import json
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def tag_translate(src, tgt, tgt_lang):
    # Input includes both task tag and language tag so model knows what to do
    inp = f"<translate> <lang:{tgt_lang}> {src}"
    return {"task": "translate", "src_lang": "en", "tgt_lang": tgt_lang, "input": inp, "target": tgt}

def tag_summarize(long, short):
    inp = f"<summarize> {long}"
    return {"task": "summarize", "input": inp, "target": short}

def tag_qa(context, question, answer):
    inp = f"<qa> Context: {context} Question: {question}"
    return {"task": "qa", "input": inp, "target": answer}

out_file = OUT / "all_tagged.jsonl"
with out_file.open("w", encoding="utf-8") as fout:
    # EN->HI
    with open(RAW / "en_hi.txt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            src, tgt = line.strip().split("\t")
            obj = tag_translate(src, tgt, "hi")
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    '''# EN->ES
    with open(RAW / "en_es.txt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            src, tgt = line.strip().split("\t")
            obj = tag_translate(src, tgt, "es")
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Summarization
    with open(RAW / "summarization.txt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            long, short = line.strip().split("\t")
            obj = tag_summarize(long, short)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # QA (context \t question \t answer)
    with open(RAW / "qa.txt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            context, question, answer = line.strip().split("\t")
            obj = tag_qa(context, question, answer)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    '''
print("Wrote:", out_file)
