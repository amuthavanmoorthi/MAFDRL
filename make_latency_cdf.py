# make_latency_cdf.py
#
# CDF of end-to-end latency T_E2E for three seeds.

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- global plotting style (match Fig. 2) ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

RESULT_DIR = "Results"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_latency(seed_dir: str) -> np.ndarray:
    csv_files = sorted(glob.glob(os.path.join(seed_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {seed_dir}")
    df = pd.read_csv(csv_files[0])

    # support both names (depending on evaluate.py version)
    if "T_e2e" in df.columns:
        col = "T_e2e"
    elif "mean_Te2e" in df.columns:
        col = "mean_Te2e"
    else:
        raise KeyError(
            f"No latency column 'T_e2e' or 'mean_Te2e' in {csv_files[0]}.\n"
            f"Columns: {list(df.columns)}"
        )

    vals = df[col].values.astype(float)
    vals = vals[np.isfinite(vals)]
    return vals


def main():
    seed_dirs = ["eval_seed0", "eval_seed1", "eval_seed2"]
    labels = ["Seed seed0", "Seed seed1", "Seed seed2"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))  # similar size as Fig. 2

    all_T = []

    for d, lab, col in zip(seed_dirs, labels, colors):
        T = load_latency(d)
        if T.size == 0:
            continue
        T_sorted = np.sort(T)
        all_T.append(T_sorted)

        N = len(T_sorted)
        cdf = np.arange(1, N + 1) / float(N)

        ax.plot(
            T_sorted,
            cdf,
            label=lab,
            linewidth=1.3,
            solid_capstyle="round",
            color=col,
        )

    if not all_T:
        raise RuntimeError("No latency data found in eval_seed* directories.")

    concat_T = np.concatenate(all_T)
    xmin = max(0.0, float(np.min(concat_T)))
    xmax = float(np.max(concat_T))
    if xmax <= xmin:
        xmax = xmin + 1e-3

    # small margin (no big white gap)
    x_margin = 0.02 * (xmax - xmin)
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("End-to-end latency $T_{\\mathrm{E2E}}$ (s)")
    ax.set_ylabel("CDF")

    ax.grid(True, linestyle="--", alpha=0.35)

    # legend inside the axes (top-left) – avoids extra bottom space
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    png_path = os.path.join(RESULT_DIR, "latency_cdf.png")
    pdf_path = os.path.join(RESULT_DIR, "latency_cdf.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"[OK] Saved: {png_path}, {pdf_path}")


if __name__ == "__main__":
    main()
