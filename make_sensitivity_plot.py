# make_sensitivity_plot.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Font and style settings
plt.rcParams["font.family"] = "Times New Roman"
sns.set(style="whitegrid")

# Define your data manually or load it from CSVs
data = {
    "CPU Frequency (GHz)": [2.0, 2.25, 2.5, 2.75, 3.0],
    "End-to-End Latency (ms)": [3.8, 3.1, 2.63, 2.58, 2.52],
    "Queueing Delay (ms)": [1.4, 1.1, 1.0, 1.0, 1.0],
    "Transmission Delay (ms)": [2.2, 1.8, 1.6, 1.5, 1.4]
}

df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["CPU Frequency (GHz)"], df["End-to-End Latency (ms)"], label="End-to-End Latency", marker="o")
ax.plot(df["CPU Frequency (GHz)"], df["Queueing Delay (ms)"], label="Queueing Delay", marker="s")
ax.plot(df["CPU Frequency (GHz)"], df["Transmission Delay (ms)"], label="Transmission Delay", marker="^")

ax.set_xlabel("UE Local CPU Frequency (GHz)")
ax.set_ylabel("Latency (ms)")
ax.legend()
plt.tight_layout()

# Save
os.makedirs("Results", exist_ok=True)
plt.savefig("Results/sensitivity_analysis.png", dpi=300)
print("[OK] Results/sensitivity_analysis.png saved")
