import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#MODEL_ID = "GenerTeam/GENERator-v2-prokaryote-3b-base"
MODEL_ID = "metaXu264/generator-v2-prokaryote-3b-atlas-ft"

# Load the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)
config = model.config

max_length = config.max_position_embeddings

# Define input sequences.
sequences = [
    "ATGAGGTGGCAAGAAATGGGCTAC",
    "GAATTCCATGAGGCTATAGAATAATCTAAGAGAAAT"
]

def left_padding(sequence, padding_char='A', multiple=6):
    remainder = len(sequence) % multiple
    if remainder != 0:
        padding_length = multiple - remainder
        return padding_char * padding_length + sequence
    return sequence

def left_truncation(sequence, multiple=6):
    remainder = len(sequence) % multiple
    if remainder != 0:
        return sequence[remainder:]
    return sequence

# Apply left_padding to all sequences
# padded_sequences = [left_padding(seq) for seq in sequences]

# Apply left_truncation to all sequences
truncated_sequences = [left_truncation(seq) for seq in sequences]

# Process the sequences
sequences = [tokenizer.bos_token + sequence for sequence in truncated_sequences]

# Tokenize the sequences
tokenizer.padding_side = "left"
inputs = tokenizer(
    sequences,
    add_special_tokens=False,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Generate the sequences
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        temperature=0.00001,
        top_k=1
    )

# Decode the generated sequences
decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the decoded sequences
print(decoded_sequences)

# It is expected to observe non-sense decoded sequences (e.g., 'AAAAAA')
# The input sequences are too short to provide sufficient context.
