import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

SEEDS = ["eval_seed0", "eval_seed1", "eval_seed2"]
OUT_PNG = os.path.join("Results", "sinr_histogram.png")
OUT_PDF = os.path.join("Results", "sinr_histogram.pdf")
os.makedirs("Results", exist_ok=True)

all_sinr = []

for d in SEEDS:
    p = os.path.join(d, "eval_metrics.csv")
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    if "mean_SINR_dB" not in df.columns:
        continue
    vals = df["mean_SINR_dB"].astype(float).dropna().values
    if vals.size > 0:
        all_sinr.append(vals)

if not all_sinr:
    raise FileNotFoundError(
        "No valid mean_SINR_dB values found in eval_seed*/eval_metrics.csv."
    )

sinr_values = np.concatenate(all_sinr, axis=0)

fig, ax = plt.subplots(figsize=(4.0, 3.0))

ax.hist(
    sinr_values,
    bins=20,
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xlabel("Post-combining SINR (dB)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.tick_params(axis="both", labelsize=9)

ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"[OK] Saved: {OUT_PNG}, {OUT_PDF}")
