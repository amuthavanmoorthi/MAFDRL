# mafdrl/experiments/plot_compression.py
import os, glob, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['figure.autolayout'] = True
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42

def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Scalars():
        return None
    sc = ea.Scalars(tag)
    df = pd.DataFrame([(x.step, x.value) for x in sc], columns=["step","value"])
    return df

def collect(root="runs"):
    rows = []
    for mode in ["none","topk","qsgd8","sign"]:
        for d in glob.glob(os.path.join(root, f"*{mode}*")):
            rwd = load_scalar(d, "train_long/round_mean_reward")
            mb  = load_scalar(d, "comm/comp_bits_total")
            den = load_scalar(d, "comm/dense_bits_total")
            if rwd is None: 
                continue
            if mb is not None and den is not None:
                # bits -> MB
                mb_sent = (mb["value"] / 8) / (1024**2)
                rows.append({
                    "mode": mode,
                    "dir": d,
                    "steps": rwd["step"].values,
                    "reward": rwd["value"].values,
                    "mb": mb_sent.values
                })
    return rows

def smooth(x, w=5):
    import numpy as np
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

if __name__ == "__main__":
    rows = collect()
    # Reward
    plt.figure(figsize=(6,4))
    for r in rows:
        y = smooth(r["reward"], w=5)
        plt.plot(r["steps"], y, label=r["mode"])
    ax = plt.gca(); ax.set_xlim(1, max([max(r["steps"]) for r in rows] + [1]))
    plt.xlabel("Federated round"); plt.ylabel("Mean reward")
    plt.legend(frameon=False)
    plt.savefig("figures/compress_reward.png", dpi=300)

    # Communication
    plt.figure(figsize=(6,4))
    for r in rows:
        y = smooth(r["mb"], w=5)
        plt.plot(r["steps"], y, label=r["mode"])
    ax = plt.gca(); ax.set_xlim(1, max([max(r["steps"]) for r in rows] + [1]))
    plt.xlabel("Federated round"); plt.ylabel("MB sent per round (global sum)")
    plt.legend(frameon=False)
    plt.savefig("figures/compress_comm.png", dpi=300)
