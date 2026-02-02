import re
from pathlib import Path

SPECIAL_TOKEN_PATH = Path(
    "/home/users/ntu/xu0029ng/scratch/models/GENERator-v2-prokaryote-3b-base/special_token.txt"
)
VOCAB_PATH = Path(
    "/home/users/ntu/xu0029ng/scratch/models/GENERator-v2-prokaryote-3b-base/vocab.txt"
)

# Match tokens like <cas1>, <crispr_spacer>, <csm3gr7>, <cas3-cas2>, <iscb-ruvciii-cterm>, etc.
# We explicitly forbid whitespace inside <> to avoid capturing lines like <total input records : ...>
TOKEN_RE = re.compile(r"<([^\s<>]+)>")

def read_vocab_tokens(vocab_path: Path) -> list[str]:
    tokens = []
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                tokens.append(t)
    return tokens

def extract_tokens_from_log(text: str) -> list[str]:
    """
    Extract <token> occurrences from the log text.
    Only keeps tokens without whitespace inside angle brackets.
    Returns deduplicated tokens preserving first-seen order.
    """
    seen = set()
    out = []
    for m in TOKEN_RE.finditer(text):
        tok = f"<{m.group(1)}>"
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out

def main():
    if not SPECIAL_TOKEN_PATH.exists():
        raise FileNotFoundError(f"special_token.txt not found: {SPECIAL_TOKEN_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"vocab.txt not found: {VOCAB_PATH}")

    # 1) Load current vocab
    vocab_tokens = read_vocab_tokens(VOCAB_PATH)
    vocab_set = set(vocab_tokens)

    # 2) Read special_token.txt (log)
    log_text = SPECIAL_TOKEN_PATH.read_text(encoding="utf-8", errors="ignore")

    # 3) Extract tokens like <cas1>
    extracted = extract_tokens_from_log(log_text)

    # 4) Decide which to append (not already in vocab)
    to_add = [t for t in extracted if t not in vocab_set]

    # 5) Append
    if to_add:
        with VOCAB_PATH.open("a", encoding="utf-8") as f:
            for t in to_add:
                f.write(t + "\n")

    # 6) Report
    print("========== VOCAB SPECIAL TOKEN APPEND SUMMARY ==========")
    print(f"special_token.txt path : {SPECIAL_TOKEN_PATH}")
    print(f"vocab.txt path         : {VOCAB_PATH}")
    print(f"Extracted tokens       : {len(extracted)}")
    print(f"Already in vocab       : {len(extracted) - len(to_add)}")
    print(f"New tokens appended    : {len(to_add)}")
    print(f"Vocab size (before)    : {len(vocab_tokens)}")
    print(f"Vocab size (after)     : {len(vocab_tokens) + len(to_add)}")

    if to_add:
        print("\nAppended tokens (in order):")
        for t in to_add:
            print(t)
    else:
        print("\nNo tokens appended: all extracted tokens already exist in vocab.txt.")

if __name__ == "__main__":
    main()
