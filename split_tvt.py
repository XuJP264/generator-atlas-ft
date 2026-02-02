#!/usr/bin/env python3
import json
import random
from pathlib import Path

print(">>> Script started")

INPUT_JSONL = Path(
    "/home/users/ntu/xu0029ng/scratch/GENERator/dataset/"
    "crispr-cas-atlas-generator/data/"
    "crispr_atlas_generator_sequences.jsonl"
)

print(f">>> Input file: {INPUT_JSONL}")
print(f">>> Exists: {INPUT_JSONL.exists()}")

OUTPUT_DIR = INPUT_JSONL.parent
print(f">>> Output dir: {OUTPUT_DIR}")
print(f">>> Output dir exists: {OUTPUT_DIR.exists()}")

data = []

with open(INPUT_JSONL, "r") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(">>> Sample line:", line[:80])
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        data.append(obj)

print(f">>> Loaded sequences: {len(data)}")

if len(data) == 0:
    raise RuntimeError("NO DATA LOADED. Input JSONL is empty or invalid.")

random.seed(42)
random.shuffle(data)

n_total = len(data)
n_train = int(n_total * 0.8)
n_valid = int(n_total * 0.1)

train = data[:n_train]
valid = data[n_train:n_train+n_valid]
test  = data[n_train+n_valid:]

def write_jsonl(path, records):
    print(f">>> Writing {path} ({len(records)})")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

write_jsonl(OUTPUT_DIR / "crispr_train.jsonl", train)
write_jsonl(OUTPUT_DIR / "crispr_valid.jsonl", valid)
write_jsonl(OUTPUT_DIR / "crispr_test.jsonl", test)

print(">>> DONE")
