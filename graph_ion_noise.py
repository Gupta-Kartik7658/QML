import json
import matplotlib.pyplot as plt

# -------------------------------------------------
# PDF font settings (journal safe)
# -------------------------------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11

# -------------------------------------------------
# Fixed color mapping
# -------------------------------------------------
FEATUREMAP_COLORS = {
    "Z": "#1f77b4",
    "ZZ": "#ff7f0e",
    "Pauli X + ZZ": "#2ca02c",
   
}

# -------------------------------------------------
# JSON paths
# -------------------------------------------------
featuremaps = {
    "Z": r"D:\qml\noise_sensitivity_ion_Pauli_X_ZZ.json",
    "ZZ": r"D:\qml\noise_sensitivity_ion_Z.json",
    "Pauli X + ZZ": r"D:\qml\noise_sensitivity_ion_ZZ.json",
    
}

# -------------------------------------------------
# Load data
# -------------------------------------------------
data = {}
for label, path in featuremaps.items():
    with open(path, "r", encoding="utf-8-sig") as f:
        data[label] = json.load(f)

sweep_names = list(next(iter(data.values()))["sweeps"].keys())

# -------------------------------------------------
# Create subplot grid
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

legend_handles = []

# -------------------------------------------------
# Plot AUC curves
# -------------------------------------------------
for idx, sweep_name in enumerate(sweep_names):
    ax = axes[idx]

    for featuremap in featuremaps.keys():
        sweep_data = data[featuremap]["sweeps"][sweep_name]

        x = [d["parameter_value"] for d in sweep_data]
        auc = [d["metrics"]["auc"] for d in sweep_data]

        line, = ax.plot(
            x,
            auc,
            marker="o",
            markersize=5,
            linewidth=2,
            color=FEATUREMAP_COLORS[featuremap],
            label=featuremap
        )

        if idx == 0:
            legend_handles.append(line)

    ax.set_xscale("log")
    ax.set_title(sweep_name, fontsize=12)
    ax.set_ylabel("AUC")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)

# -------------------------------------------------
# Hide unused subplot
# -------------------------------------------------
axes[-1].axis("off")

# -------------------------------------------------
# Global legend
# -------------------------------------------------
fig.legend(
    handles=legend_handles,
    labels=[h.get_label() for h in legend_handles],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=12
)

output_path = r"D:\qml\noise_sensitivity_trapped_ion.pdf"

plt.savefig(
    output_path,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.show()