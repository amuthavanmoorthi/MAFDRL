import os, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150

SEEDS = ["eval_seed0", "eval_seed1", "eval_seed2"]
OUT = os.path.join("Results", "offload_distribution.png")
os.makedirs("Results", exist_ok=True)

dfs = []
for d in SEEDS:
    p = os.path.join(d, "eval_metrics.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p)
        df["seed"] = d
        dfs.append(df)
if not dfs:
    raise FileNotFoundError("No eval_seed*/eval_metrics.csv found.")

df = pd.concat(dfs, ignore_index=True)

# gather all rho columns
rho_cols = [c for c in df.columns if c.startswith("rho_u")]
U = len(rho_cols)

# Build a long-form table for box/violin plots
long = []
for u in range(U):
    col = f"rho_u{u}"
    tmp = df[[col]].copy()
    tmp["UE"] = f"UE{u+1}"
    tmp.rename(columns={col: "rho"}, inplace=True)
    long.append(tmp)
long = pd.concat(long, ignore_index=True)

fig, ax = plt.subplots(figsize=(6.0, 3.8))
# violin plot (nice for distributions)
parts = ax.violinplot(
    [long[long["UE"]==f"UE{i+1}"]["rho"].values for i in range(U)],
    showmedians=True
)
ax.set_xticks(np.arange(1, U+1))
ax.set_xticklabels([f"UE{i+1}" for i in range(U)])
ax.set_ylabel("Offloading ratio ρ")
ax.set_xlabel("User equipment (UE)")
ax.set_ylim(0, 1)

# put legend-like caption under the plot
plt.subplots_adjust(bottom=0.25)
ax.text(0.5, -0.25, "Distribution of offloading ratios across seeds (no title per IEEE style)",
        transform=ax.transAxes, ha="center", va="top")

plt.savefig(OUT, bbox_inches="tight")
print(f"[OK] {OUT}")
