import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#MODEL_ID = "GenerTeam/GENERator-v2-prokaryote-3b-base"
MODEL_ID = "metaXu264/generator-v2-prokaryote-3b-atlas-ft"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)


# Get model configuration
config = model.config
max_length = config.max_position_embeddings

# Define input sequences
sequences = [
    "ATGAGGTGGCAAGAAATGGGCTAC",
    "GAATTCCATGAGGCTATAGAATAATCTAAGAGAAAT"
]

# Truncate each sequence to the nearest multiple of 6
processed_sequences = [
    tokenizer.bos_token + seq[:len(seq)//6*6]
    for seq in sequences
]

# Tokenization
tokenizer.padding_side = "right"
inputs = tokenizer(
    processed_sequences,
    add_special_tokens=True,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Model Inference
with torch.inference_mode():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states[-1]
attention_mask = inputs["attention_mask"]

# Option 1: Last token (EOS) embedding
last_token_indices = attention_mask.sum(dim=1) - 1
eos_embeddings = hidden_states[
    torch.arange(hidden_states.size(0)),
    last_token_indices,
    :
]

# Option 2: Mean pooling over all tokens
expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(torch.float32)
sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
mean_embeddings = sum_embeddings / expanded_mask.sum(dim=1)

# Output
print("EOS (Last Token) Embeddings:", eos_embeddings)
print("Mean Pooling Embeddings:", mean_embeddings)

# ============================================================================
# Additional notes:
# - The preprocessing step ensures sequences are multiples of 6 for 6-mer tokenizer
# - For causal LM, the last token embedding (EOS) is commonly used
# - Mean pooling considers all tokens including BOS and content tokens
# - The choice depends on your downstream task requirements
# - Both methods handle variable sequence lengths via attention mask
# ============================================================================
