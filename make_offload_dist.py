# make_offload_dist.py
#
# Offloading ratio distribution per UE (UE1..UE3).

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

RESULT_DIR = "Results"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_rho_from_dir(seed_dir: str):
    csv_files = sorted(glob.glob(os.path.join(seed_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {seed_dir}")
    df = pd.read_csv(csv_files[0])

    # assumes evaluate.py saved rho_u0, rho_u1, rho_u2
    r0 = df["rho_u0"].values.astype(float)
    r1 = df["rho_u1"].values.astype(float)
    r2 = df["rho_u2"].values.astype(float)
    return r0, r1, r2


def main():
    seed_dirs = ["eval_seed0", "eval_seed1", "eval_seed2"]

    rho_all = {0: [], 1: [], 2: []}
    for d in seed_dirs:
        r0, r1, r2 = load_rho_from_dir(d)
        rho_all[0].extend(r0.tolist())
        rho_all[1].extend(r1.tolist())
        rho_all[2].extend(r2.tolist())

    data = [np.array(rho_all[0]), np.array(rho_all[1]), np.array(rho_all[2])]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=["UE1", "UE2", "UE3"],
        widths=0.45,
        whis=1.5,
        showfliers=False,
    )

    # soft colors
    box_colors = ["#c6dbef", "#9ecae1", "#6baed6"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_linewidth(0.9)

    # overall mean per UE (green triangle)
    means = [float(np.mean(d)) for d in data]
    x = np.array([1, 2, 3], dtype=float)
    ax.scatter(
        x,
        means,
        s=45,
        marker="^",
        color="green",
        edgecolor="black",
        linewidth=0.5,
        label="Overall mean",
        zorder=5,
    )

    ax.set_ylabel("Offloading ratio $\\rho$")
    ax.set_xlabel("User equipment (UE)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, 3.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # legend inside plot, top-right – no bottom margin
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    png_path = os.path.join(RESULT_DIR, "offload_distribution.png")
    pdf_path = os.path.join(RESULT_DIR, "offload_distribution.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"[OK] Saved: {png_path}, {pdf_path}")


if __name__ == "__main__":
    main()
