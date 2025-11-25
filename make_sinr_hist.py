import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("eval_seed2/eval_metrics.csv")

df_clean = df.dropna(subset=["mean_SINR_dB"])
sinr_values = df_clean["mean_SINR_dB"]

if not sinr_values.empty:
    plt.figure(figsize=(6, 4))
    plt.hist(sinr_values, bins=20, color='black', edgecolor='white')
    plt.xlabel("Post-combining SINR (dB)", fontname="Times New Roman", fontsize=12)
    plt.ylabel("Count", fontname="Times New Roman", fontsize=12)
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")
    plt.tight_layout()
    os.makedirs("Results", exist_ok=True)
    plt.savefig("Results/sinr_histogram.png", dpi=300)
    plt.savefig("Results/sinr_histogram.pdf", dpi=300)
    print("[OK] Saved: Results/sinr_histogram.png")
else:
    print("[WARN] No valid SINR data found.")
