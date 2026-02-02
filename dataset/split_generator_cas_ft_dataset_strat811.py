#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# Config
# =========================
INPUT_PATH = Path("/home/users/ntu/xu0029ng/scratch/GENERator/dataset/generator_cas_ft_dataset.jsonl")
OUT_DIR = INPUT_PATH.parent

SEED = 42
RATIO = {"train": 0.8, "valid": 0.1, "test": 0.1}

# output filenames (same directory)
OUT_TRAIN = OUT_DIR / "train.jsonl"
OUT_VALID = OUT_DIR / "valid.jsonl"
OUT_TEST  = OUT_DIR / "test.jsonl"

PLOT_CAT = OUT_DIR / "category_distribution.png"
PLOT_SPLIT_CAT = OUT_DIR / "category_split_distribution.png"
PLOT_TOP_TAGS = OUT_DIR / "top_tags.png"

TOPK_TAGS = 30  # top N fine-grained tags to visualize


# =========================
# Helpers
# =========================
TAG_RE = re.compile(r"^<([^>]+)>", re.IGNORECASE)

def extract_tag(seq: str) -> str:
    """Return the raw tag inside <...>, or 'unknown' if missing."""
    m = TAG_RE.match(seq)
    if not m:
        return "unknown"
    return m.group(1)

def coarse_category(tag: str) -> str:
    """
    Map many fine tags to 4 coarse categories (case-insensitive):
      spacer / repeat / tracer / other
    'tracer' here corresponds to tracrRNA-like tags (contains 'tracr').
    """
    t = tag.lower()
    if "spacer" in t:
        return "spacer"
    if "repeat" in t:
        return "repeat"
    if "tracr" in t:   # tracrRNA / tracrrna / tracr...
        return "tracer"
    return "other"

def compute_targets(n: int):
    """Exact integer targets for 8/1/1 split with remainder to test."""
    n_train = int(n * RATIO["train"])
    n_valid = int(n * RATIO["valid"])
    n_test = n - n_train - n_valid
    return {"train": n_train, "valid": n_valid, "test": n_test}

def choose_split(remain: dict, rng: random.Random):
    """
    Choose split proportional to remaining quotas (guarantees exact totals).
    remain: dict with keys train/valid/test and nonnegative ints.
    """
    total = remain["train"] + remain["valid"] + remain["test"]
    if total <= 0:
        return None
    r = rng.randrange(total)
    if r < remain["train"]:
        return "train"
    r -= remain["train"]
    if r < remain["valid"]:
        return "valid"
    return "test"


# =========================
# Pass 1: statistics
# =========================
rng = random.Random(SEED)

fine_counts = Counter()
coarse_counts = Counter()

print(f"[Pass1] Scanning for stats: {INPUT_PATH}")
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seq = obj.get("sequence", "")
        tag = extract_tag(seq)
        cat = coarse_category(tag)
        fine_counts[tag] += 1
        coarse_counts[cat] += 1
        if i % 2_000_000 == 0:
            print(f"  processed {i:,} lines...")

print("\n=== Coarse category counts (4-way) ===")
for k in ["spacer", "repeat", "tracer", "other"]:
    print(f"{k:7s}: {coarse_counts.get(k, 0):,}")

print("\n=== Top fine-grained tags ===")
for tag, cnt in fine_counts.most_common(20):
    print(f"{tag:20s}: {cnt:,}")

# visualize coarse distribution
plt.figure(figsize=(7, 5))
cats = ["spacer", "repeat", "tracer", "other"]
vals = [coarse_counts.get(c, 0) for c in cats]
plt.bar(cats, vals)
plt.title("Coarse category distribution (full dataset)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_CAT, dpi=200)
plt.close()

# visualize top fine tags
top_tags = fine_counts.most_common(TOPK_TAGS)
plt.figure(figsize=(10, 6))
plt.bar([t for t, _ in top_tags], [c for _, c in top_tags])
plt.title(f"Top {TOPK_TAGS} fine-grained tags")
plt.ylabel("Count")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(PLOT_TOP_TAGS, dpi=200)
plt.close()


# =========================
# Prepare per-category targets
# =========================
targets = {cat: compute_targets(coarse_counts[cat]) for cat in cats}
remain = {cat: targets[cat].copy() for cat in cats}

print("\n=== Per-category split targets (exact) ===")
for cat in cats:
    t = targets[cat]
    print(f"{cat:7s}: train={t['train']:,}  valid={t['valid']:,}  test={t['test']:,}")

# =========================
# Pass 2: split with exact quotas (streaming)
# =========================
split_counts = {s: Counter() for s in ["train", "valid", "test"]}

print(f"\n[Pass2] Writing splits to:\n  {OUT_TRAIN}\n  {OUT_VALID}\n  {OUT_TEST}")
with INPUT_PATH.open("r", encoding="utf-8") as fin, \
     OUT_TRAIN.open("w", encoding="utf-8") as ftr, \
     OUT_VALID.open("w", encoding="utf-8") as fva, \
     OUT_TEST.open("w", encoding="utf-8") as fte:

    for i, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seq = obj.get("sequence", "")
        tag = extract_tag(seq)
        cat = coarse_category(tag)

        # choose split based on remaining quota for this category
        s = choose_split(remain[cat], rng)
        if s is None:
            # Should not happen; but if it does, fall back to test.
            s = "test"

        if s == "train":
            ftr.write(json.dumps(obj, ensure_ascii=False) + "\n")
        elif s == "valid":
            fva.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            fte.write(json.dumps(obj, ensure_ascii=False) + "\n")

        remain[cat][s] -= 1
        split_counts[s][cat] += 1

        if i % 2_000_000 == 0:
            print(f"  processed {i:,} lines...")

# sanity check: all quotas should be zero
print("\n=== Remaining quotas (should be 0) ===")
for cat in cats:
    print(f"{cat:7s}: {remain[cat]}")

print("\n=== Split coarse counts ===")
for s in ["train", "valid", "test"]:
    print(f"[{s}] " + "  ".join(f"{cat}={split_counts[s][cat]:,}" for cat in cats))

# visualize split-wise coarse distribution (grouped bars)
plt.figure(figsize=(9, 6))
x = range(len(cats))
width = 0.25
for idx, s in enumerate(["train", "valid", "test"]):
    counts = [split_counts[s][cat] for cat in cats]
    plt.bar([xi + idx * width for xi in x], counts, width=width, label=s)
plt.xticks([xi + width for xi in x], cats)
plt.ylabel("Count")
plt.title("Coarse category distribution after stratified 8/1/1 split")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_SPLIT_CAT, dpi=200)
plt.close()

print("\nDone.")
print(f"Plots:\n  {PLOT_CAT}\n  {PLOT_SPLIT_CAT}\n  {PLOT_TOP_TAGS}")
