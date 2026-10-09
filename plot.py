import json
import matplotlib.pyplot as plt

# ---------------------------
# Load JSON data
# ---------------------------
with open("noise_sensitivity_ZZ.json", "r") as f:
    data = json.load(f)

sweeps = data["sweeps"]

# Expected 7 sweep keys (example order preserved from JSON)
sweep_names = list(sweeps.keys())

# ---------------------------
# Create 3x3 subplot grid
# ---------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# ---------------------------
# Plot each sweep
# ---------------------------
for idx, sweep_name in enumerate(sweep_names):
    ax = axes[idx]
    sweep_data = sweeps[sweep_name]

    x = [item["parameter_value"] for item in sweep_data]
    y = [item["metrics"]["far"] for item in sweep_data]

    ax.plot(x, y, marker="o")
    ax.set_title(sweep_name)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Accuracy")
    ax.grid(True)

# ---------------------------
# Hide unused subplots (9 - 7 = 2)
# ---------------------------
for i in range(len(sweep_names), len(axes)):
    axes[i].axis("off")

# ---------------------------
# Layout & display
# ---------------------------
plt.tight_layout()
plt.show()
