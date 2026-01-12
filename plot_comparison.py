import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Define the competitive ratios
def cr_single_agent_v_known(v):
    """Single-agent CR when v is known: 1 + (v + sqrt(2 - v^2)) / (1 - v)"""
    return 1 + (v + np.sqrt(2 - v**2)) / (1 - v)

def cr_single_agent_v_unknown(v):
    """Single-agent CR when v is unknown: 1 + 2(1 + v) / (1 - v)"""
    return 1 + 2 * (1 + v) / (1 - v)

def cr_split_up_case1(v):
    """Split-up Case 1 (fast agent catches): (3 + v) / (1 - v)"""
    return (3 + v) / (1 - v)

def cr_split_up_case2(u, v):
    """Split-up Case 2 (slow agent catches): 2(1 - v) / (u(1 + u)(u - v))"""
    # Add small epsilon to avoid division by zero near u = v
    denom = u * (1 + u) * (u - v)
    result = np.where(np.abs(denom) > 1e-10, 2 * (1 - v) / denom, np.inf)
    return result

# Create grid with higher resolution
u_vals = np.linspace(0.001, 0.999, 800)
v_vals = np.linspace(0.001, 0.999, 800)
U, V = np.meshgrid(u_vals, v_vals)

# Only consider Case 2 when u is sufficiently greater than v (to avoid numerical issues near boundary)
margin = 0.01  # small margin to avoid numerical instability
case2_valid = U > V + margin

# Split-up CR: use Case 2 when u > v, otherwise Case 1
CR_split_case1 = cr_split_up_case1(V)
CR_split_case2 = np.where(case2_valid, cr_split_up_case2(U, V), np.inf)

# The split-up algorithm uses Case 2 when u > v (and the slow agent catches first)
# Otherwise it defaults to Case 1
CR_split = np.where(case2_valid, np.minimum(CR_split_case1, CR_split_case2), CR_split_case1)

# Single-agent CRs
CR_single_known = cr_single_agent_v_known(V)
CR_single_unknown = cr_single_agent_v_unknown(V)

# Regions where split-up is better
split_up_better_known = CR_split < CR_single_known
split_up_better_unknown = CR_split < CR_single_unknown

# Create single plot (v known case only)
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the region where split-up is better
ax.contourf(U, V, split_up_better_known.astype(int), levels=[0.5, 1.5], colors=['#2ecc71'], alpha=0.7)
ax.contour(U, V, split_up_better_known.astype(int), levels=[0.5], colors=['#27ae60'], linewidths=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='$u = v$')
ax.set_xlabel('$u$ (slow agent speed)', fontsize=12)
ax.set_ylabel('$v$ (fugitive speed)', fontsize=12)
ax.set_title('Region where Split-Up Algorithm Outperforms Single-Agent Algorithm', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='#27ae60', linewidth=2, label='Split-up better'),
    plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='$u = v$')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('FIG/split_up_comparison.pdf', bbox_inches='tight')
plt.savefig('FIG/split_up_comparison.png', bbox_inches='tight', dpi=150)
print("Saved figures to FIG/split_up_comparison.pdf and FIG/split_up_comparison.png")
plt.show()
