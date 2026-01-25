#!/usr/bin/env python3
import json
import sys
from pathlib import Path

from tqdm import tqdm

# =========================
# Configuration
# =========================

INPUT_JSON = "crispr-cas-atlas-v1.0.json"
OUTPUT_JSONL = "crispr_atlas_generator_sequences.jsonl"

VALID_BASES = set("ACGT")

# =========================
# Utility functions
# =========================

def normalize_dna(seq: str) -> str:
    """
    Normalize sequence to DNA:
    - Uppercase
    - RNA U -> T
    - Remove non-ACGT characters
    """
    seq = seq.upper().replace("U", "T")
    return "".join(c for c in seq if c in VALID_BASES)


def pad_to_multiple_of_six(seq: str) -> str:
    """
    LEFT padding with 'A' to ensure length is a multiple of 6.
    """
    r = len(seq) % 6
    if r != 0:
        seq = "A" * (6 - r) + seq
    return seq


# =========================
# Record processing
# =========================

def extract_sequence(record: dict) -> str | None:
    """
    Extract a single DNA sequence from a CRISPR-Cas Atlas operon record.
    """
    parts = []

    # -------- CRISPR arrays --------
    crispr_blocks = record.get("crispr") or []
    for crispr in crispr_blocks:
        if not isinstance(crispr, dict):
            continue

        repeat = crispr.get("crispr_repeat")
        if repeat:
            parts.append(normalize_dna(repeat))

        spacers = crispr.get("crispr_spacers") or []
        for spacer in spacers:
            if spacer:
                parts.append(normalize_dna(spacer))

    # -------- tracrRNA --------
    tracr_block = record.get("tracr")
    if isinstance(tracr_block, dict):
        tracr_seq = tracr_block.get("tracr")
        if tracr_seq:
            parts.append(normalize_dna(tracr_seq))

    if not parts:
        return None

    seq = "".join(parts)
    return pad_to_multiple_of_six(seq)


# =========================
# Main
# =========================

def main():
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSONL)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"📖 Loading CRISPR-Cas Atlas: {input_path}")
    with input_path.open("r") as f:
        data = json.load(f)

    total = len(data)
    print(f"🧬 Total operon records: {total}")

    kept = 0
    skipped = 0

    with output_path.open("w") as out:
        for record in tqdm(
            data,
            total=total,
            desc="Processing operons",
            unit="operon",
            dynamic_ncols=True,
        ):
            try:
                seq = extract_sequence(record)
            except Exception:
                skipped += 1
                continue

            if seq is None or len(seq) < 6:
                skipped += 1
                continue

            out.write(json.dumps({"sequence": seq}) + "\n")
            kept += 1

    print("\n✅ Processing completed")
    print(f"✅ Sequences written: {kept}")
    print(f"⚠️  Records skipped: {skipped}")
    print(f"📦 Output file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
