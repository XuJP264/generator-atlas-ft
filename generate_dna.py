import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= 配置 =========
MODEL_DIR = "results/atlas_ft_3b/checkpoint-36000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 输入序列（必须 6 的倍数）======
# 示例：36 bp = 6 * 6
prompt = "ATGCGTATGCGTATGCGTATGCGTATGCGTATGCGT"

assert len(prompt) % 6 == 0, "Input length must be multiple of 6"

# ========= 加载 tokenizer / model =========
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
).to(DEVICE)

model.eval()

# ========= Tokenize =========
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(DEVICE)

# ========= Generation =========
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,          # 生成长度（bp ≈ tokens * 6）
        do_sample=True,              # 必须采样
        temperature=0.8,             # 0.7–1.0 推荐
        top_p=0.95,
        repetition_penalty=1.1,      # 防止重复
        eos_token_id=None            # DNA 没有 EOS
    )

# ========= Decode =========
generated = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n===== PROMPT =====")
print(prompt)

print("\n===== GENERATED =====")
print(generated)
print(f"\nGenerated length: {len(generated)} bp")
