# # mafdrl/eval_plots.py
# import os, glob, csv
# import numpy as np
# import torch as th
# import matplotlib.pyplot as plt
# from scipy.special import expit

# from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
# from mafdrl.agents.maddpg import MADDPG

# def load_actors(algo: MADDPG, ckpt_dir="checkpoints"):
#     paths = sorted(glob.glob(os.path.join(ckpt_dir, "actor_agent*.pt")))
#     if not paths:
#         print("No checkpoints found; run training first.")
#         return False
#     sds = [th.load(p, map_location="cpu") for p in paths]
#     algo.set_actor_params(sds)
#     print(f"Loaded {len(sds)} actor checkpoints from {ckpt_dir}")
#     return True

# def eval_and_plot(U=3, Mt=2, Nr=4, steps=500, seed=123, outdir="."):
#     os.makedirs(outdir, exist_ok=True)

#     # Env + algo
#     env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
#     obs_dim = 4
#     act_dim = 2 + Mt + 1
#     algo = MADDPG(U, obs_dim, act_dim, device="cpu")
#     _ = load_actors(algo)

#     # Data buffers
#     mean_rewards = []
#     all_sinr = []
#     all_Te2e = []
#     all_Tloc = []
#     all_Ttx = []
#     all_TQ = []
#     all_Tcpu = []

#     # Rollout (deterministic: noise_scale=0)
#     obs = env.reset()
#     for t in range(steps):
#         acts = algo.act(obs, noise_scale=0.0)
#         # map actions to valid ranges
#         acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)     # p
#         acts[:, 1] = expit(acts[:, 1])                              # rho
#         # w left as-is; env normalizes inside
#         # f_loc with stability floor (same as train)
#         f_min = 0.10 * env.f_loc_max
#         acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

#         obs, rew, done, trunc, info = env.step(acts)

#         mean_rewards.append(float(np.mean(rew)))
#         all_sinr.append(info["sinr"].copy())
#         all_Te2e.append(info["T_e2e"].copy())
#         all_Tloc.append(info["T_loc"].copy())
#         all_Ttx.append(info["T_tx"].copy())
#         all_TQ.append(info["T_Q"].copy())
#         all_Tcpu.append(info["T_cpu"].copy())

#     # Convert to arrays
#     mean_rewards = np.asarray(mean_rewards)                   # (steps,)
#     sinr = np.vstack(all_sinr)                                # (steps,U)
#     Te2e = np.vstack(all_Te2e)
#     Tloc = np.vstack(all_Tloc)
#     Ttx  = np.vstack(all_Ttx)
#     TQ   = np.vstack(all_TQ)
#     Tcpu = np.vstack(all_Tcpu)

#     # ---- Save CSV for paper tables ----
#     csv_path = os.path.join(outdir, "eval_metrics.csv")
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["step", "mean_reward", "mean_Te2e", "mean_Tloc", "mean_Ttx", "mean_TQ", "mean_Tcpu", "mean_SINR_dB"])
#         for i in range(len(mean_rewards)):
#             mean_sinr_db = 10*np.log10(max(1e-12, np.mean(sinr[i])))
#             w.writerow([
#                 i+1,
#                 f"{mean_rewards[i]:.6f}",
#                 f"{np.mean(Te2e[i]):.6e}",
#                 f"{np.mean(Tloc[i]):.6e}",
#                 f"{np.mean(Ttx[i]):.6e}",
#                 f"{np.mean(TQ[i]):.6e}",
#                 f"{np.mean(Tcpu[i]):.6e}",
#                 f"{mean_sinr_db:.4f}",
#             ])
#     print(f"Saved CSV: {csv_path}")

#     # ---- Plot 1: reward curve ----
#     plt.figure()
#     plt.plot(np.arange(1, len(mean_rewards)+1), mean_rewards)
#     plt.xlabel("Step")
#     plt.ylabel("Mean reward")
#     # plt.title("Evaluation: Mean Reward vs Step")
#     plt.tight_layout()
#     path1 = os.path.join(outdir, "eval_reward_curve.png")
#     plt.savefig(path1, dpi=150)
#     print(f"Saved plot: {path1}")

#     # ---- Plot 2: SINR histogram (dB) ----
#     sinr_db = 10*np.log10(np.clip(sinr.flatten(), 1e-12, None))
#     plt.figure()
#     plt.hist(sinr_db, bins=40)
#     plt.xlabel("SINR (dB)")
#     plt.ylabel("Count")
#     # plt.title("Evaluation: SINR Distribution (dB)")
#     plt.tight_layout()
#     path2 = os.path.join(outdir, "eval_sinr_hist.png")
#     plt.savefig(path2, dpi=150)
#     print(f"Saved plot: {path2}")

#     # ---- Plot 3: E2E latency histogram ----
#     plt.figure()
#     plt.hist(Te2e.flatten(), bins=40)
#     plt.xlabel("T_E2E (s)")
#     plt.ylabel("Count")
#     # plt.title("Evaluation: End-to-End Latency Distribution")
#     plt.tight_layout()
#     path3 = os.path.join(outdir, "eval_latency_hist.png")
#     plt.savefig(path3, dpi=150)
#     print(f"Saved plot: {path3}")

#     # ---- Plot 4: Latency components over time (averaged across users) ----
#     plt.figure()
#     plt.plot(np.mean(Tloc, axis=1), label="T_loc")
#     plt.plot(np.mean(Ttx,  axis=1), label="T_tx")
#     plt.plot(np.mean(TQ,   axis=1), label="T_Q")
#     plt.plot(np.mean(Tcpu, axis=1), label="T_cpu")
#     plt.xlabel("Step")
#     plt.ylabel("Latency (s)")
#     # plt.title("Evaluation: Average Latency Components vs Step")
#     plt.legend()
#     plt.tight_layout()
#     path4 = os.path.join(outdir, "eval_latency_components.png")
#     plt.savefig(path4, dpi=150)
#     print(f"Saved plot: {path4}")

#     print("Done.")

# if __name__ == "__main__":
#     # Change steps if you want longer curves
#     eval_and_plot(steps=500, outdir=".")

# mafdrl/eval_plots.py
import os, glob, csv
import numpy as np
import torch as th
from scipy.special import expit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.maddpg import MADDPG


# =========================
# IEEE plot style (Windows-safe Times New Roman)
# =========================
def _set_ieee_style():
    # Try to register Times New Roman explicitly on Windows (common internal names)
    tnr_candidates = [
        r"C:\Windows\Fonts\times.ttf",              # regular
        r"C:\Windows\Fonts\timesbd.ttf",            # bold
        r"C:\Windows\Fonts\Times New Roman.ttf",    # sometimes present
        r"C:\Windows\Fonts\Times.ttf",              # fallback
    ]
    for fp in tnr_candidates:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass  # ignore if already registered / unsupported

    # Set serif family with TNR first and safe fallbacks
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif']

    # Sizes & figure polish
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['figure.dpi'] = 300

    # Remove outer whitespace on save
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0
    mpl.rcParams['figure.autolayout'] = True

    # Keep text editable if you export to PDF
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

_set_ieee_style()


def load_actors(algo: MADDPG, ckpt_dir="checkpoints"):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "actor_agent*.pt")))
    if not paths:
        print("No checkpoints found; run training first.")
        return False
    sds = [th.load(p, map_location="cpu") for p in paths]
    algo.set_actor_params(sds)
    print(f"Loaded {len(sds)} actor checkpoints from {ckpt_dir}")
    return True


def _tighten_axes(ax, xdata=None, ydata=None):
    """Remove inner whitespace; clamp to data with a tiny pad."""
    ax.margins(x=0, y=0)
    if xdata is not None and len(xdata):
        ax.set_xlim(np.min(xdata), np.max(xdata))
    if ydata is not None and len(ydata):
        ymin, ymax = np.min(ydata), np.max(ydata)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            rng = ymax - ymin
            pad = 0.01 * rng
            ax.set_ylim(ymin - pad, ymax + pad)


def eval_and_plot(U=3, Mt=2, Nr=4, steps=500, seed=123, outdir="."):
    os.makedirs(outdir, exist_ok=True)

    # Env + algo
    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    obs_dim = 4
    act_dim = 2 + Mt + 1
    algo = MADDPG(U, obs_dim, act_dim, device="cpu")
    _ = load_actors(algo)

    # Data buffers
    mean_rewards = []
    all_sinr = []
    all_Te2e = []
    all_Tloc = []
    all_Ttx = []
    all_TQ = []
    all_Tcpu = []

    # Rollout (deterministic: noise_scale=0)
    obs = env.reset()
    for t in range(steps):
        acts = algo.act(obs, noise_scale=0.0)
        # map actions to valid ranges
        acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)     # p
        acts[:, 1] = expit(acts[:, 1])                              # rho
        # w left as-is; env normalizes inside
        # f_loc with stability floor (same as train)
        f_min = 0.10 * env.f_loc_max
        acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

        obs, rew, done, trunc, info = env.step(acts)

        mean_rewards.append(float(np.mean(rew)))
        all_sinr.append(info["sinr"].copy())
        all_Te2e.append(info["T_e2e"].copy())
        all_Tloc.append(info["T_loc"].copy())
        all_Ttx.append(info["T_tx"].copy())
        all_TQ.append(info["T_Q"].copy())
        all_Tcpu.append(info["T_cpu"].copy())

    # Convert to arrays
    mean_rewards = np.asarray(mean_rewards)                   # (steps,)
    sinr = np.vstack(all_sinr)                                # (steps,U)
    Te2e = np.vstack(all_Te2e)
    Tloc = np.vstack(all_Tloc)
    Ttx  = np.vstack(all_Ttx)
    TQ   = np.vstack(all_TQ)
    Tcpu = np.vstack(all_Tcpu)

    # ---- Save CSV for paper tables ----
    csv_path = os.path.join(outdir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "mean_Te2e", "mean_Tloc", "mean_Ttx", "mean_TQ", "mean_Tcpu", "mean_SINR_dB"])
        for i in range(len(mean_rewards)):
            mean_sinr_db = 10*np.log10(max(1e-12, np.mean(sinr[i])))
            w.writerow([
                i+1,
                f"{mean_rewards[i]:.6f}",
                f"{np.mean(Te2e[i]):.6e}",
                f"{np.mean(Tloc[i]):.6e}",
                f"{np.mean(Ttx[i]):.6e}",
                f"{np.mean(TQ[i]):.6e}",
                f"{np.mean(Tcpu[i]):.6e}",
                f"{mean_sinr_db:.4f}",
            ])
    print(f"Saved CSV: {csv_path}")

    # Common x for time-series plots
    xs = np.arange(1, len(mean_rewards) + 1)

    # ---- Plot 1: reward curve ----
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(xs, mean_rewards, lw=1.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean reward")
    _tighten_axes(ax, xs, mean_rewards)
    path1 = os.path.join(outdir, "eval_reward_curve.png")
    fig.savefig(path1, dpi=300)  # bbox/pad handled by rcParams
    plt.close(fig)
    print(f"Saved plot: {path1}")

    # ---- Plot 2: SINR histogram (dB) ----
    sinr_db = 10*np.log10(np.clip(sinr.flatten(), 1e-12, None))
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    counts, bins, _ = ax.hist(sinr_db, bins=40)
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("")
    # Tighten to bin edges; remove inner margins
    ax.set_xlim(bins[0], bins[-1])
    _tighten_axes(ax)
    path2 = os.path.join(outdir, "eval_sinr_hist.png")
    fig.savefig(path2, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path2}")

    # ---- Plot 3: E2E latency histogram ----
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    counts, bins, _ = ax.hist(Te2e.flatten(), bins=40)
    ax.set_xlabel("T_E2E (s)")
    ax.set_ylabel("")
    ax.set_xlim(bins[0], bins[-1])
    _tighten_axes(ax)
    path3 = os.path.join(outdir, "eval_latency_hist.png")
    fig.savefig(path3, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path3}")

    # ---- Plot 4: Latency components over time (averaged across users) ----
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(xs, np.mean(Tloc, axis=1), label="T_loc", lw=1.4)
    ax.plot(xs, np.mean(Ttx,  axis=1), label="T_tx",  lw=1.4)
    ax.plot(xs, np.mean(TQ,   axis=1), label="T_Q",   lw=1.4)
    ax.plot(xs, np.mean(Tcpu, axis=1), label="T_cpu", lw=1.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (s)")
    ax.legend(frameon=False)
    _tighten_axes(ax, xs, np.hstack([
        np.mean(Tloc, axis=1),
        np.mean(Ttx,  axis=1),
        np.mean(TQ,   axis=1),
        np.mean(Tcpu, axis=1)
    ]))
    path4 = os.path.join(outdir, "eval_latency_components.png")
    fig.savefig(path4, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path4}")
    print("Done.")

if __name__ == "__main__":
    # Change steps if you want longer curves
    eval_and_plot(steps=500, outdir=".")
