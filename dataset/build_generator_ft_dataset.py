import json
import re

INPUT_PATH = "/home/users/ntu/xu0029ng/scratch/GENERator/dataset/extracted_sequences.jsonl"
OUTPUT_PATH = "generator_cas_ft_dataset.jsonl"

def clean_sequence(seq: str) -> str:
    """
    Clean DNA/RNA sequence:
    - Uppercase
    - Convert U -> T (RNA -> DNA alphabet)
    - Remove non-ATCG characters
    """
    seq = seq.upper()
    seq = seq.replace("U", "T")
    seq = re.sub(r"[^ATCG]", "", seq)
    return seq

def pad_to_multiple_of_6(seq: str) -> str:
    """
    Pad sequence with 'A' to make its length a multiple of 6
    """
    remainder = len(seq) % 6
    if remainder != 0:
        seq += "A" * (6 - remainder)
    return seq

def get_special_token(obj) -> str:
    """
    Rule:
    1. If cas_gene exists and is non-empty -> <cas_gene>
    2. Otherwise -> <sequence_role>
    """
    cas_gene = obj.get("cas_gene", None)
    if cas_gene is not None:
        cas_gene = str(cas_gene).strip()
        if cas_gene != "":
            return f"<{cas_gene.lower()}>"

    sequence_role = obj.get("sequence_role", "unknown")
    return f"<{str(sequence_role).lower()}>"

def main():
    n_total = 0
    n_kept = 0

    with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
        for line in fin:
            n_total += 1
            obj = json.loads(line)

            # ===== filtering =====
            if obj.get("sequence_type") not in {"dna", "rna"}:
                continue
            if "sequence" not in obj:
                continue

            # ===== build sequence =====
            special_token = get_special_token(obj)
            seq = clean_sequence(obj["sequence"])
            seq = pad_to_multiple_of_6(seq)

            if len(seq) == 0:
                continue

            final_seq = special_token + seq
            fout.write(json.dumps({"sequence": final_seq}) + "\n")
            n_kept += 1

    print("========== DATASET BUILD SUMMARY ==========")
    print(f"Total input lines     : {n_total}")
    print(f"Kept for fine-tuning  : {n_kept}")
    print(f"Output dataset path  : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
