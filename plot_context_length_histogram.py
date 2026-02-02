#!/usr/bin/env python3
"""
Plot distribution of sequence lengths in crispr_train.jsonl

Bins:
0–192, 192–384, ..., >1920

Pure matplotlib, HPC-safe
"""

import json
import os
import matplotlib.pyplot as plt

# =========================
# 0. Paths
# =========================

data_dir = "/home/users/ntu/xu0029ng/scratch/GENERator/dataset/crispr-cas-atlas-generator/data"
jsonl_path = os.path.join(data_dir, "crispr_train.jsonl")
output_path = os.path.join(data_dir, "context_length_distribution.png")

# =========================
# 1. Read JSONL & collect lengths
# =========================

lengths = []

with open(jsonl_path, "r") as f:
    for line_num, line in enumerate(f, 1):
        record = json.loads(line)

        # Common field names (robust handling)
        if "sequence" in record:
            seq = record["sequence"]
        elif "text" in record:
            seq = record["text"]
        else:
            raise KeyError(
                f"Line {line_num}: no 'sequence' or 'text' field found"
            )

        lengths.append(len(seq))

print(f"Loaded {len(lengths)} sequences")

# =========================
# 2. Define bins
# =========================

bin_edges = list(range(0, 1921, 192)) + [float("inf")]
bin_labels = []

for i in range(len(bin_edges) - 1):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    if right == float("inf"):
        bin_labels.append(">1920")
    else:
        bin_labels.append(f"{left}-{right}")

# =========================
# 3. Count per bin
# =========================

counts = [0] * (len(bin_edges) - 1)

for L in lengths:
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= L < bin_edges[i + 1]:
            counts[i] += 1
            break

# =========================
# 4. Plot bar chart
# =========================

plt.figure(figsize=(10, 5))

plt.bar(
    bin_labels,
    counts
)

plt.xlabel("Context Length (sequence length)")
plt.ylabel("Number of sequences")
plt.title("Distribution of Sequence Context Lengths")

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print("Saved histogram to:", output_path)
