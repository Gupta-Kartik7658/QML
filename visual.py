import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Setup the Figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')  # Hide axes

# Helper Function to draw styled boxes
def draw_box(x, y, title, subtitle, content, color_edge, color_face):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.1, y-0.1), 3, 4, boxstyle="round,pad=0.1", 
                                    ec="none", fc='gray', alpha=0.3)
    ax.add_patch(shadow)
    
    # Main Box
    box = patches.FancyBboxPatch((x, y), 3, 4, boxstyle="round,pad=0.1", 
                                 ec=color_edge, fc=color_face, linewidth=2)
    ax.add_patch(box)
    
    # Header Background
    header_box = patches.FancyBboxPatch((x, y+3.2), 3, 0.8, boxstyle="round,pad=0.1", 
                                        ec="none", fc=color_edge, alpha=0.1)
    # Clip the bottom of the header to make it look like a header (optional simple overlay)
    ax.add_patch(header_box)
    
    # Title & Subtitle
    ax.text(x + 1.5, y + 3.7, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#333')
    ax.text(x + 1.5, y + 3.4, subtitle, ha='center', va='center', fontsize=9, style='italic', color='#555')
    
    # Content (Equations & Text)
    # We use explicit newlines and LaTeX math between $ signs
    ax.text(x + 1.5, y + 1.5, content, ha='center', va='center', fontsize=10, linespacing=1.8)

# --- Define Content based on your Paper ---

# Box 1: Data Preprocessing
content_1 = (
    "Input: 30 PCA Features\n"
    r"$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$" "\n"
    r"Undersampling $1:1$"
)
draw_box(0.5, 1, "1. Data Preprocessing", "(Credit Card DB)", content_1, "#64748b", "#f8fafc")

# Arrow 1
ax.annotate("", xy=(4.2, 3), xytext=(3.7, 3), arrowprops=dict(arrowstyle="->", lw=2, color="#94a3b8"))

# Box 2: QGFS Optimizer
content_2 = (
    "Greedy Loop\n"
    r"$S^* = \mathrm{arg\,max}_{S \subset F} \text{AUC}(S)$" "\n"
    r"Best: ZZMap + SC"
)
draw_box(4.5, 1, "2. QGFS Optimizer", "(Feature Selection)", content_2, "#3b82f6", "#eff6ff")

# Arrow 2
ax.annotate("", xy=(8.2, 3), xytext=(7.7, 3), arrowprops=dict(arrowstyle="->", lw=2, color="#94a3b8"))

# Box 3: Quantum Kernel
content_3 = (
    r"$|\psi(\mathbf{x})\rangle = \mathcal{U}_{\phi}(\mathbf{x})|0\rangle$" "\n"
    r"$K_{ij} = |\langle \psi_i | \psi_j \rangle|^2$" "\n"
    "Fidelity Test"
)
draw_box(8.5, 1, "3. Quantum Kernel", "(Hilbert Space)", content_3, "#6366f1", "#eef2ff")

# Arrow 3
ax.annotate("", xy=(12.2, 3), xytext=(11.7, 3), arrowprops=dict(arrowstyle="->", lw=2, color="#94a3b8"))

# Box 4: QSVM
content_4 = (
    "Dual Optimization\n"
    r"$f(x) = \text{sgn}(\sum \alpha_i y_i K + b)$" "\n"
    r"AUC: $0.9920$"
)
draw_box(12.5, 1, "4. QSVM", "(Classification)", content_4, "#22c55e", "#f0fdf4")

# Title
plt.suptitle("Quantum Kernel-Based Financial Fraud Detection Methodology", fontsize=14, fontweight='bold', y=0.95)

# Save
plt.tight_layout()
plt.savefig("methodology_diagram.png", dpi=300, bbox_inches='tight')
plt.show()