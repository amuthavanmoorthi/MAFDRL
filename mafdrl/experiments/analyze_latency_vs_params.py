# mafdrl/experiments/analyze_latency_vs_params.py
import os, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.special import expit
import torch as th

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.maddpg import MADDPG

# ============== IEEE-style plotting (Times New Roman; Windows safe) ==============
def set_ieee_style():
    for fp in [r"C:\Windows\Fonts\times.ttf",
               r"C:\Windows\Fonts\timesbd.ttf",
               r"C:\Windows\Fonts\Times New Roman.ttf",
               r"C:\Windows\Fonts\Times.ttf"]:
        if os.path.exists(fp):
            try: font_manager.fontManager.addfont(fp)
            except: pass
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0
    mpl.rcParams['figure.autolayout'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

set_ieee_style()

# ---------- Helpers ----------
def load_mafdrl_policy_adapt(U_target, Mt, device="cpu", ckpt_dir="checkpoints_long"):
    """
    Load MA-FDRL checkpoints and adapt to any target U:
      - if U_target > U_trained: tile actors
      - if U_target < U_trained: slice first U_target
    Returns policy_fn(obs)->acts with correct (U_target, act_dim).
    """
    # read all saved actors
    sd_paths = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("actor_agent") and f.endswith(".pt")])
    if not sd_paths:
        raise FileNotFoundError(f"No actor checkpoints found in {ckpt_dir}")
    sds_src = [th.load(p, map_location=device) for p in sd_paths]
    U_tr = len(sds_src)

    obs_dim, act_dim = 4, 2 + Mt + 1
    algo = MADDPG(U_target, obs_dim, act_dim, device=device)

    # build list of state dicts of length U_target by tiling/slicing
    sds_tgt = [sds_src[i % U_tr] for i in range(U_target)]
    algo.set_actor_params(sds_tgt)

    def policy_fn(obs):
        return algo.act(obs, noise_scale=0.0)
    return policy_fn

def eval_latency(policy_fn, U, Mt, Nr, pmax, steps=500, seed=123):
    """Evaluate average E2E latency under given parameters."""
    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    env.p_max = float(pmax)  # override transmit power cap
    obs = env.reset()
    all_Te2e = []
    for _ in range(steps):
        acts = policy_fn(obs)
        # bound actions the same way as training
        acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)   # power
        acts[:, 1] = expit(acts[:, 1])                           # rho
        f_min = 0.10 * env.f_loc_max
        acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)
        obs, rew, done, trunc, info = env.step(acts)
        all_Te2e.append(np.mean(info["T_e2e"]))
    return float(np.mean(all_Te2e)), float(np.std(all_Te2e))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_long", help="Path to MA-FDRL actor checkpoints")
    ap.add_argument("--outdir", default="figures", help="Output folder")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--Nr", type=int, default=4)
    ap.add_argument("--Mt", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # -------- (1) Latency vs Transmit Power --------
    powers = np.linspace(0.1, 1.0, 6)  # 0.1W → 1.0W
    lat_mean, lat_std = [], []

    # Use U equal to #actors in checkpoints (typical: 3)
    # but read from the directory to be safe
    trained_U = len([f for f in os.listdir(args.ckpt) if f.startswith("actor_agent") and f.endswith(".pt")]) or 3

    for p in powers:
        policy = load_mafdrl_policy_adapt(U_target=trained_U, Mt=args.Mt, device=device, ckpt_dir=args.ckpt)
        mean_Te2e, std_Te2e = eval_latency(policy, U=trained_U, Mt=args.Mt, Nr=args.Nr,
                                           pmax=p, steps=args.steps)
        lat_mean.append(mean_Te2e); lat_std.append(std_Te2e)
        print(f"[Power {p:.2f} W] mean_Te2e = {mean_Te2e:.4f} s")

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(powers, lat_mean, yerr=lat_std, marker='o', capsize=3, label="MA-FDRL")
    ax.set_xlabel("User transmit power $P_u^{\\max}$ (W)")
    ax.set_ylabel("Mean end-to-end latency $T_{\\sf E2E}$ (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False); ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "latency_vs_power.png"), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved -> {os.path.join(args.outdir, 'latency_vs_power.png')}")

    # -------- (2) Latency vs Number of Users --------
    users = [2, 3, 4, 5, 6]
    lat_mean2, lat_std2 = [], []

    for U in users:
        # adapt checkpoints to U (tile/slice)
        policy = load_mafdrl_policy_adapt(U_target=U, Mt=args.Mt, device=device, ckpt_dir=args.ckpt)
        mean_Te2e, std_Te2e = eval_latency(policy, U=U, Mt=args.Mt, Nr=args.Nr,
                                           pmax=0.5, steps=args.steps)
        lat_mean2.append(mean_Te2e); lat_std2.append(std_Te2e)
        print(f"[Users {U}] mean_Te2e = {mean_Te2e:.4f} s")

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(users, lat_mean2, yerr=lat_std2, marker='o', color='C1', capsize=3, label="MA-FDRL")
    ax.set_xlabel("Number of users $U$")
    ax.set_ylabel("Mean end-to-end latency $T_{\\sf E2E}$ (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False); ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "latency_vs_users.png"), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved -> {os.path.join(args.outdir, 'latency_vs_users.png')}")

    print("Analysis complete.")

if __name__ == "__main__":
    main()
