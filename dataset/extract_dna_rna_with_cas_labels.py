import json
import argparse
from pathlib import Path


def clean_seq(seq: str) -> str:
    if not seq:
        return None
    return seq.replace("\n", "").replace(" ", "").upper()


# ========= Dataset 1 =========
# cas_proteins_with_dna.jsonl
def process_cas_proteins_with_dna(input_path, fout):
    with open(input_path, "r") as f:
        for line in f:
            obj = json.loads(line)

            dna = clean_seq(obj.get("predicted_dna"))
            if dna is None:
                continue

            record = {
                "sequence_type": "dna",
                "cas_gene": obj.get("gene_name"),      # Cas9 / Cas1 / Cas10 ...
                "sequence_role": "cas_protein",
                "sequence": dna
            }

            fout.write(json.dumps(record) + "\n")


# ========= Dataset 2 =========
# crispr-cas-atlas-v1.0.json
def process_crispr_cas_atlas(input_path, fout):
    with open(input_path, "r") as f:
        data = json.load(f)

    for operon in data:

        # ---- CRISPR repeats & spacers ----
        for crispr in operon.get("crispr", []):
            repeat = clean_seq(crispr.get("crispr_repeat"))
            if repeat:
                fout.write(json.dumps({
                    "sequence_type": "dna",
                    "cas_gene": None,
                    "sequence_role": "crispr_repeat",
                    "sequence": repeat
                }) + "\n")

            for spacer in crispr.get("crispr_spacers", []):
                spacer = clean_seq(spacer)
                if spacer:
                    fout.write(json.dumps({
                        "sequence_type": "dna",
                        "cas_gene": None,
                        "sequence_role": "crispr_spacer",
                        "sequence": spacer
                    }) + "\n")

        # ---- tracrRNA (FIXED) ----
        tracr_block = operon.get("tracr")
        if isinstance(tracr_block, dict):
            tracr = clean_seq(tracr_block.get("tracr"))
        else:
            tracr = None

        if tracr:
            fout.write(json.dumps({
                "sequence_type": "rna",
                "cas_gene": None,
                "sequence_role": "tracrRNA",
                "sequence": tracr
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas_protein_jsonl", required=True)
    parser.add_argument("--crispr_atlas_json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as fout:
        process_cas_proteins_with_dna(args.cas_protein_jsonl, fout)
        process_crispr_cas_atlas(args.crispr_atlas_json, fout)

    print(f"[DONE] Output written to {args.output}")


if __name__ == "__main__":
    main()
