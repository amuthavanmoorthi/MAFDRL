import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150

SEEDS = ["eval_seed0", "eval_seed1", "eval_seed2"]
OUT = os.path.join("Results", "latency_cdf.png")
os.makedirs("Results", exist_ok=True)

def derive_te2e(df: pd.DataFrame) -> np.ndarray:
    # Prefer mean_Te2e if present and non-NaN
    if "mean_Te2e" in df.columns:
        arr = df["mean_Te2e"].astype(float).dropna().values
        if arr.size > 0:
            return arr
    # Fallback: sum components (seconds)
    need = ["mean_Ttx", "mean_TQ", "mean_Tcpu", "mean_Tloc"]
    if all(c in df.columns for c in need):
        comp = df[need].astype(float)
        arr = comp.sum(axis=1).dropna().values
        if arr.size > 0:
            return arr
    return np.array([])

all_lat = []
labels = []
for d in SEEDS:
    p = os.path.join(d, "eval_metrics.csv")
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    te2e = derive_te2e(df)
    if te2e.size == 0:
        continue
    all_lat.append(np.sort(te2e))
    labels.append(d)

if not all_lat:
    raise FileNotFoundError("No usable latency series found. "
                            "Ensure eval_seed*/eval_metrics.csv have either mean_Te2e or all components.")

# Decide units: if typical values < 0.1 sec, show ms
typical = float(np.mean(np.concatenate(all_lat)))
use_ms = typical < 0.1
xlabel = r"End-to-end latency $T_{\mathrm{E2E}}$ (ms)" if use_ms else r"End-to-end latency $T_{\mathrm{E2E}}$ (s)"

fig, ax = plt.subplots(figsize=(6.0, 3.8))
for arr, lab in zip(all_lat, labels):
    x = arr * 1e3 if use_ms else arr
    y = np.linspace(0, 1, len(x), endpoint=True)
    ax.plot(x, y, lw=1.8, label=lab.replace("_", " "))

ax.set_xlabel(xlabel)
ax.set_ylabel("CDF")
ax.grid(True, alpha=0.3)

# place legend below the plot, no title on top
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=len(labels), frameon=False)
plt.subplots_adjust(bottom=0.28)

plt.savefig(OUT, bbox_inches="tight")
print(f"[OK] {OUT}")
