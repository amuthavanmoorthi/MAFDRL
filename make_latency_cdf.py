import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- CONFIG ----
SEED_FILES = [
    "mafdrl-project/eval_seed0/eval_metrics.csv",
    "mafdrl-project/eval_seed1/eval_metrics.csv",
    "mafdrl-project/eval_seed2/eval_metrics.csv",
]
OUT_DIR = "Results"
URRLC_THRESHOLD_MS = 10.0     # threshold line in ms
LAT_COL = "mean_Te2e"         # column in CSV (seconds)
# -----------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---- FONT SETTINGS: Times New Roman ----
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 0.8

# ---- Load data ----
dfs = []
for p in SEED_FILES:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    if LAT_COL not in df.columns:
        raise KeyError(f"Column '{LAT_COL}' not found in {p}. "
                       f"Available: {list(df.columns)}")
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
lat_ms = all_df[LAT_COL].to_numpy() * 1000.0  # seconds → ms

# ---- Compute CDF ----
lat_sorted = np.sort(lat_ms)
cdf = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)

# ---- Statistics ----
mean_ms = np.mean(lat_ms)
std_ms = np.std(lat_ms)
p_below_10 = np.mean(lat_ms <= URRLC_THRESHOLD_MS)

print("========== Latency CDF summary ==========")
print(f"Samples: {len(lat_ms)}")
print(f"Mean T^E2E (ms): {mean_ms:.2f}")
print(f"Std  T^E2E (ms): {std_ms:.2f}")
print(f"P[T^E2E <= {URRLC_THRESHOLD_MS:.0f} ms]: {p_below_10*100:.2f}%")
print("=========================================")

# ---- Plot ----
plt.figure(figsize=(6,4))
plt.plot(lat_sorted, cdf, linewidth=2, color='black')
plt.axvline(URRLC_THRESHOLD_MS, linestyle='--', linewidth=1, color='gray')

plt.xlabel(r'End-to-end latency $T^{\mathrm{E2E}}$ (ms)', labelpad=4)
plt.ylabel('CDF', labelpad=4)

# No title
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
plt.tight_layout()

out_png = os.path.join(OUT_DIR, "latency_cdf_ieee.png")
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print(f"Saved high-resolution figure: {out_png}")
print("✅ Figure ready for IEEE paper (Times New Roman, no title).")