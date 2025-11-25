import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150

SEEDS = ["eval_seed0", "eval_seed1", "eval_seed2"]
OUT = os.path.join("Results", "latency_components_bar.png")
os.makedirs("Results", exist_ok=True)

metrics = ["mean_Ttx", "mean_TQ", "mean_Tcpu", "mean_Tloc"]
labels = ["Tx", "Queue", "CPU (BS)", "Local"]

seed_avgs = []
for d in SEEDS:
    p = os.path.join(d, "eval_metrics.csv")
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    m = df[metrics].mean().values  # seconds
    seed_avgs.append(m)
if not seed_avgs:
    raise FileNotFoundError("No eval_seed*/eval_metrics.csv found.")

arr = np.vstack(seed_avgs)  # shape: (num_seeds, 4)
means = arr.mean(axis=0)
stds  = arr.std(axis=0)

# convert to ms if small
scale_label = "(ms)" if means.mean()<0.1 else "(s)"
scale = 1e3 if scale_label=="(ms)" else 1.0
means *= scale; stds *= scale

fig, ax = plt.subplots(figsize=(6.0, 3.8))
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel(f"Latency {scale_label}")
ax.set_xlabel("Component")
ax.grid(axis="y", alpha=0.3)

plt.subplots_adjust(bottom=0.22)
ax.text(0.5, -0.22, "Mean ± std across seeds; components from info dict per step",
        transform=ax.transAxes, ha="center", va="top")

plt.savefig(OUT, bbox_inches="tight")
print(f"[OK] {OUT}")
