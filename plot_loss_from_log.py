import re
import ast
import matplotlib.pyplot as plt

log_path = "generator_atlas_ft_3b_4gpu_1epoch.log"

losses = []
epochs = []
steps = []

step = 0

pattern = re.compile(r"\{.*'loss':.*\}")

with open(log_path, "r") as f:
    for line in f:
        if pattern.search(line):
            try:
                data = ast.literal_eval(pattern.search(line).group())
                losses.append(data["loss"])
                epochs.append(data.get("epoch", None))
                steps.append(step)
                step += 1
            except Exception:
                pass

print(f"Parsed {len(losses)} loss points")

# ===== Plot =====
plt.figure(figsize=(8, 5))
plt.plot(steps, losses, label="Train Loss")
plt.xlabel("Logging Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("loss_curve.png", dpi=200)
plt.show()
