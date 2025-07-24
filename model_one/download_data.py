from datasets import load_dataset
import json

dataset = load_dataset("Kevew/test", split="train")

with open("data.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        json.dump(example, f)
        f.write("\n")