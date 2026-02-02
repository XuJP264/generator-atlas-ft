#!/usr/bin/env python3
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================

MODEL_ID_1 = "GenerTeam/GENERator-v2-prokaryote-3b-base"
MODEL_ID_2 = "metaXu264/generator-v2-prokaryote-3b-atlas-ft"
TEST_JSONL = (
    "/home/users/ntu/xu0029ng/scratch/GENERator/"
    "dataset/crispr-cas-atlas-generator/data/"
    "crispr_test.jsonl"
)

CONTEXT_LEN = 768     # number of bp used as context
K_MER = 6            # predict next K bp
MAX_SAMPLES = None   # e.g. 10000 for quick test

OUTPUT_TXT = (
    "/home/users/ntu/xu0029ng/scratch/GENERator/"
    "results/next_kmer_eval_generator_v2_atlas_base.txt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utilities
# =========================

def left_truncation(sequence, multiple=6):
    r = len(sequence) % multiple
    if r != 0:
        return sequence[r:]
    return sequence


def count_lines(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)

# =========================
# Load model
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
).to(DEVICE)

model.eval()
config = model.config
max_length = config.max_position_embeddings

# =========================
# Count total lines
# =========================

TOTAL_LINES = count_lines(TEST_JSONL)

print(f"Total lines in test file: {TOTAL_LINES}")

# =========================
# Evaluation loop
# =========================

total = 0        # valid evaluated samples
correct = 0

with open(TEST_JSONL, "r") as f:
    pbar = tqdm(
        f,
        total=TOTAL_LINES,
        desc="Evaluating (lines)",
        unit="lines",
        dynamic_ncols=True
    )

    for idx, line in enumerate(pbar):
        if MAX_SAMPLES is not None and idx >= MAX_SAMPLES:
            break

        obj = json.loads(line)
        seq = obj["sequence"].upper()

        # ensure enough length
        if len(seq) < CONTEXT_LEN + K_MER:
            continue

        # truncate to tokenizer multiple
        seq = left_truncation(seq, multiple=6)

        context = seq[:CONTEXT_LEN]
        target_kmer = seq[CONTEXT_LEN:CONTEXT_LEN + K_MER]

        # prepare input
        input_seq = tokenizer.bos_token + context
        tokenizer.padding_side = "left"

        inputs = tokenizer(
            input_seq,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(DEVICE)

        # generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=K_MER,
                temperature=1e-5,
                top_k=1,
                do_sample=False
            )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        generated_kmer = decoded[len(context):len(context) + K_MER]

        if generated_kmer == target_kmer:
            correct += 1

        total += 1

        # update progress info every 100 valid samples
        if total % 100 == 0:
            pbar.set_postfix(
                evaluated=total,
                acc=f"{correct/total:.4f}"
            )

# =========================
# Report
# =========================

accuracy = correct / total if total > 0 else 0.0

report_lines = [
    "========== Next-K-mer Generation Evaluation ==========",
    f"Model        : {MODEL_ID}",
    f"Test set     : {TEST_JSONL}",
    f"Context len  : {CONTEXT_LEN}",
    f"K-mer        : {K_MER}",
    f"Samples      : {total}",
    f"Correct      : {correct}",
    f"Accuracy     : {accuracy:.6f}",
]

report_text = "\n".join(report_lines)

print("\n" + report_text)

# save to txt
with open(OUTPUT_TXT, "w") as f:
    f.write(report_text + "\n")

print(f"\nResults saved to: {OUTPUT_TXT}")
