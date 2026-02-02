#!/usr/bin/env python3
import json
import torch
import csv
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================

TEST_JSONL = (
    "/home/users/ntu/xu0029ng/scratch/GENERator/"
    "dataset/crispr-cas-atlas-generator/data/"
    "crispr_test.jsonl"
)

MODEL_IDS = {
    "base": "GenerTeam/GENERator-v2-prokaryote-3b-base",
    "ft":   "metaXu264/generator-v2-prokaryote-3b-atlas-ft",
}

CONTEXT_LIST = [96, 192, 288, 384, 480, 576, 672, 768, 864, 960]
K_MER = 6
MAX_SAMPLES = 10000

RESULTS_CSV = (
    "/home/users/ntu/xu0029ng/scratch/GENERator/results/"
    "next_kmer_context_sweep.csv"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utilities
# =========================

def left_truncation(sequence, multiple=6):
    r = len(sequence) % multiple
    return sequence[r:] if r != 0 else sequence


def load_test_sequences(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("sequence") or obj.get("text")
            if seq is not None:
                seqs.append(seq.upper())
    return seqs


def init_csv_if_needed(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "context_len", "samples", "correct", "accuracy"]
            )
            writer.writeheader()


def append_result(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "context_len", "samples", "correct", "accuracy"]
        )
        writer.writerow(row)

# =========================
# Init
# =========================

print("Initializing CSV (incremental saving enabled)...")
init_csv_if_needed(RESULTS_CSV)

print("Loading test sequences...")
ALL_SEQS = load_test_sequences(TEST_JSONL)
print(f"Total sequences loaded: {len(ALL_SEQS)}")

# =========================
# Main evaluation
# =========================

for model_name, model_id in MODEL_IDS.items():
    print(f"\n===== Loading model: {model_name} =====")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id
    ).to(DEVICE)
    model.eval()

    max_length = model.config.max_position_embeddings

    for context_len in CONTEXT_LIST:
        print(f"\n[Model: {model_name}] Context = {context_len}")

        total = 0
        correct = 0

        pbar = tqdm(
            ALL_SEQS,
            desc=f"{model_name} | ctx={context_len}",
            dynamic_ncols=True
        )

        for seq in pbar:
            if total >= MAX_SAMPLES:
                break

            if len(seq) < context_len + K_MER:
                continue

            seq = left_truncation(seq, multiple=6)
            context = seq[:context_len]
            target = seq[context_len:context_len + K_MER]

            input_seq = tokenizer.bos_token + context
            tokenizer.padding_side = "left"

            inputs = tokenizer(
                input_seq,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(DEVICE)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=K_MER,
                    temperature=1e-5,
                    top_k=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            pred = decoded[len(context):len(context) + K_MER]

            if pred == target:
                correct += 1

            total += 1

            if total % 200 == 0:
                pbar.set_postfix(acc=f"{correct/total:.4f}")

        acc = correct / total if total > 0 else 0.0
        print(f"Finished: model={model_name}, ctx={context_len}, "
              f"acc={acc:.6f}, N={total}")

        # =========================
        # Incremental save (CRITICAL)
        # =========================
        row = {
            "model": model_name,
            "context_len": context_len,
            "samples": total,
            "correct": correct,
            "accuracy": acc,
        }
        append_result(RESULTS_CSV, row)

        print(f"Saved result to CSV: {row}")

print("\nAll finished. Results are safely saved incrementally.")
print(f"CSV path: {RESULTS_CSV}")
