import json
import matplotlib.pyplot as plt

# -------------------------------------------------
# PDF font settings (IMPORTANT for journals)
# -------------------------------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11

# -------------------------------------------------
# Fixed color mapping (consistent across all panels)
# -------------------------------------------------
featuremaps = {
    "Z": {
        "path": r"D:\qml\noise_sensitivity_Z.json",
        "color": "#1f77b4",  # blue
    },
    "ZZ": {
        "path": r"D:\qml\noise_sensitivity_ZZ.json",
        "color": "#ff7f0e",  # orange
    },
    "Pauli X + ZZ": {
        "path": r"D:\qml\noise_sensitivity_Pauli_X_ZZ.json",
        "color": "#2ca02c",  # green
    },
}

# -------------------------------------------------
# Load JSON data
# -------------------------------------------------
data = {}
for label, cfg in featuremaps.items():
    with open(cfg["path"], "r", encoding="utf-8-sig") as f:
        data[label] = json.load(f)

# Noise parameters (assumed identical across feature maps)
sweep_names = list(next(iter(data.values()))["sweeps"].keys())

# -------------------------------------------------
# Create subplot grid
# -------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

legend_handles = []

# -------------------------------------------------
# Plot AUC curves
# -------------------------------------------------
for idx, sweep_name in enumerate(sweep_names):
    ax = axes[idx]

    for featuremap, cfg in featuremaps.items():
        sweep_data = data[featuremap]["sweeps"][sweep_name]

        x = [d["parameter_value"] for d in sweep_data]
        auc = [d["metrics"]["auc"] for d in sweep_data]

        line, = ax.plot(
            x,
            auc,
            marker="o",
            linewidth=2,
            markersize=5,
            color=cfg["color"],
            label=featuremap
        )

        # Collect legend handles once
        if idx == 0:
            legend_handles.append(line)

    ax.set_xscale("log")
    ax.set_title(sweep_name, fontsize=12)
    ax.set_ylabel("AUC")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)

# -------------------------------------------------
# Hide unused subplots
# -------------------------------------------------
for i in range(len(sweep_names), len(axes)):
    axes[i].axis("off")

# -------------------------------------------------
# Global legend (paper standard)
# -------------------------------------------------
fig.legend(
    handles=legend_handles,
    labels=[h.get_label() for h in legend_handles],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=12
)

# -------------------------------------------------
# Layout adjustment
# -------------------------------------------------
# plt.tight_layout(rect=[0, 0.08, 1, 1])

# -------------------------------------------------
# Save high-quality PDF (VECTOR FORMAT)
# -------------------------------------------------
output_path = r"D:\qml\noise_sc_sensitivity_analysis.pdf"

plt.savefig(
    output_path,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

# -------------------------------------------------
# Show figure
# -------------------------------------------------
plt.show()
