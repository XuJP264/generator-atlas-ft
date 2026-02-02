import json
from collections import Counter
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/users/ntu/xu0029ng/scratch/GENERator/dataset/extracted_sequences.jsonl",
        help="Path to extracted_sequences.jsonl"
    )
    args = parser.parse_args()

    role_counter = Counter()
    type_counter = Counter()
    role_type_counter = Counter()
    total = 0

    with open(args.input, "r") as f:
        for line in f:
            obj = json.loads(line)
            total += 1

            role = obj.get("sequence_role")
            seq_type = obj.get("sequence_type")

            role_counter[role] += 1
            type_counter[seq_type] += 1
            role_type_counter[(role, seq_type)] += 1

    print("\n========== SUMMARY ==========")
    print(f"Total sequences: {total}")

    print("\n--- sequence_role counts ---")
    for k, v in role_counter.most_common():
        print(f"{k:15s} : {v}")

    print("\n--- sequence_type counts ---")
    for k, v in type_counter.most_common():
        print(f"{k:5s} : {v}")

    print("\n--- (sequence_role, sequence_type) counts ---")
    for (role, seq_type), v in role_type_counter.most_common():
        print(f"{role:15s} | {seq_type:3s} : {v}")


if __name__ == "__main__":
    main()
