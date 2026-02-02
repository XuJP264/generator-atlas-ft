import json

JSONL_PATH = "/home/users/ntu/xu0029ng/scratch/GENERator/dataset/crispr-cas-atlas-generator/data/crispr_test.jsonl"
LENGTH_THRESHOLD = 1920/2

total = 0
longer_than_threshold = 0
max_len = 0
invalid_lines = 0

with open(JSONL_PATH, "r") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue

        # 兼容不同字段名
        seq = obj.get("sequence") or obj.get("text")
        if seq is None:
            continue

        seq_len = len(seq)
        total += 1
        max_len = max(max_len, seq_len)

        if seq_len > LENGTH_THRESHOLD:
            longer_than_threshold += 1

print("====== Sequence Length Statistics ======")
print(f"Total valid sequences       : {total}")
print(f"Sequences > {LENGTH_THRESHOLD} bp : {longer_than_threshold}")
print(f"Max sequence length         : {max_len}")
print(f"Invalid JSON lines skipped  : {invalid_lines}")
