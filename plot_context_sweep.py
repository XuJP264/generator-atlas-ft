#!/usr/bin/env python3
"""
Visualization for next_kmer_context_sweep.csv

Outputs:
1) Split-panel figure (ft / base)
2) Shared-axis figure (log-scale y)

Pure matplotlib, HPC-safe
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0. Paths
# =========================

csv_path = "/home/users/ntu/xu0029ng/scratch/GENERator/results/next_kmer_context_sweep.csv"
output_dir = "/home/users/ntu/xu0029ng/scratch/GENERator/results"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. Load data
# =========================

df = pd.read_csv(csv_path)

# Ensure ordering by context length
df = df.sort_values("context_len")

df_base = df[df["model"] == "base"]
df_ft   = df[df["model"] == "ft"]

# =========================
# 2. Figure 1: Split panels
# =========================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 7),
    sharex=True
)

# ---- ft model (top) ----
axes[0].plot(
    df_ft["context_len"],
    df_ft["accuracy"],
    marker="o",
    linewidth=2
)
axes[0].set_ylabel("Accuracy (ft)")
axes[0].set_title("Next k-mer Prediction Accuracy vs Context Length")
axes[0].grid(alpha=0.3)

# ---- base model (bottom) ----
axes[1].plot(
    df_base["context_len"],
    df_base["accuracy"],
    marker="o",
    linewidth=2
)
axes[1].set_xlabel("Context Length (tokens)")
axes[1].set_ylabel("Accuracy (base)")
axes[1].grid(alpha=0.3)

# x-axis settings (explicit, linear)
axes[1].set_xlim(96, 960)
axes[1].set_xticks(df_base["context_len"])

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "accuracy_vs_context_len_split.png"),
    dpi=300
)
plt.close()

# =========================
# 3. Figure 2: Shared axis
#    (linear x, log y)
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    df_base["context_len"],
    df_base["accuracy"],
    marker="o",
    linewidth=2,
    label="base"
)

plt.plot(
    df_ft["context_len"],
    df_ft["accuracy"],
    marker="o",
    linewidth=2,
    label="finetuned"
)

plt.xlabel("Context Length (tokens)")
plt.ylabel("Accuracy")
plt.yscale("log")  # only y-axis
plt.xlim(96, 960)
plt.xticks(df_base["context_len"])

plt.title("Next k-mer Prediction Accuracy vs Context Length")
plt.legend()
plt.grid(which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "accuracy_vs_context_len_shared_logy.png"),
    dpi=300
)
plt.close()

print("Figures saved to:", output_dir)
