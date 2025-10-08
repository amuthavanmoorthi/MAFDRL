# mafdrl/experiments/analyze_mimo_scaling.py
import os, argparse, math
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


# ------------------------- Helpers -------------------------
def count_trained_users(ckpt_dir):
    return len([f for f in os.listdir(ckpt_dir) if f.startswith("actor_agent") and f.endswith(".pt")])

def load_mafdrl_policy(U_target, Mt_src, device="cpu", ckpt_dir="checkpoints_long"):
    """Load trained MA-FDRL actors (saved with Mt_src) and tile/slice to U_target users."""
    sd_paths = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                       if f.startswith("actor_agent") and f.endswith(".pt")])
    if not sd_paths:
        raise FileNotFoundError(f"No actor checkpoints found in {ckpt_dir}")
    sds_src = [th.load(p, map_location=device) for p in sd_paths]
    U_tr = len(sds_src)
    obs_dim = 4
    act_dim_src = 2 + Mt_src + 1  # what the trained nets output

    # Build a dummy MADDPG with U_target and the *source* act_dim (Mt_src) to run actors
    from mafdrl.agents.maddpg import MADDPG
    algo_src = MADDPG(U_target, obs_dim, act_dim_src, device=device)
    sds_tgt = [sds_src[i % U_tr] for i in range(U_target)]
    algo_src.set_actor_params(sds_tgt)
    return algo_src  # we will adapt its actions to Mt_target below


def adapt_beam_logits(acts_src, Mt_src, Mt_tgt):
    """
    Map actor outputs with Mt_src beam logits to Mt_tgt required by the env:
      - keep p, rho as-is
      - adapt w logits by tiling (Mt_tgt > Mt_src) or truncating (Mt_tgt < Mt_src)
      - keep f_loc as last column
    """
    if Mt_src == Mt_tgt:
        return acts_src

    p = acts_src[:, [0]]
    rho = acts_src[:, [1]]
    w_src = acts_src[:, 2:2+Mt_src]
    f_loc = acts_src[:, [-1]]

    if Mt_tgt > Mt_src:
        reps = math.ceil(Mt_tgt / Mt_src)
        w_tiled = np.tile(w_src, (1, reps))[:, :Mt_tgt]
        w_tgt = w_tiled
    else:  # Mt_tgt < Mt_src
        w_tgt = w_src[:, :Mt_tgt]  # simple truncation (works well in practice)

    return np.concatenate([p, rho, w_tgt, f_loc], axis=1)


def rollout_eval_MtNr(U, Mt_target, Nr_target, algo_src, Mt_src, steps=500, seed=123, pmax=None):
    """Run one evaluation with (Mt_target, Nr_target) using source actors trained with Mt_src."""
    env = MECURLLCEnv(U=U, Mt=Mt_target, Nr=Nr_target, seed=seed)
    if pmax is not None:
        env.p_max = float(pmax)
    obs = env.reset()

    Te2e, SINR = [], []
    for _ in range(steps):
        acts_src = algo_src.act(obs, noise_scale=0.0)  # shape (U, 2+Mt_src+1)
        acts = adapt_beam_logits(acts_src, Mt_src=Mt_src, Mt_tgt=Mt_target)

        # enforce bounds (same as elsewhere)
        acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)   # power
        acts[:, 1] = expit(acts[:, 1])                           # rho
        f_min = 0.10 * env.f_loc_max
        acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

        obs, rew, done, trunc, info = env.step(acts)
        Te2e.append(np.mean(info["T_e2e"]))
        SINR.append(np.mean(info["sinr"]))

    Te2e = np.asarray(Te2e)
    SINR = 10*np.log10(np.clip(np.asarray(SINR), 1e-12, None))
    return float(Te2e.mean()), float(Te2e.std()), float(SINR.mean()), float(SINR.std())


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_long", help="MA-FDRL checkpoints directory")
    ap.add_argument("--U", type=int, default=3, help="Number of users to evaluate")
    ap.add_argument("--Mt-src", type=int, default=2, help="Mt used during training/saving")
    ap.add_argument("--Mt-list", type=int, nargs="+", default=[1,2,4,8], help="Mt sweep")
    ap.add_argument("--Nr-list", type=int, nargs="+", default=[1,2,4,8], help="Nr sweep")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--pmax", type=float, default=None, help="Override p_max if provided")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # load trained actors (with Mt_src) and adapt per (Mt,Nr)
    algo_src = load_mafdrl_policy(U_target=args.U, Mt_src=args.Mt_src, device=device, ckpt_dir=args.ckpt)

    # ---- 1) Latency/SINR vs Mt (fix Nr = median of list) ----
    Nr_fix = sorted(args.Nr_list)[len(args.Nr_list)//2]
    mt_mean, mt_std, mt_sinr_m, mt_sinr_s = [], [], [], []
    for Mt in args.Mt_list:
        m, s, sm, ss = rollout_eval_MtNr(args.U, Mt, Nr_fix, algo_src, args.Mt_src,
                                         steps=args.steps, seed=123, pmax=args.pmax)
        mt_mean.append(m); mt_std.append(s); mt_sinr_m.append(sm); mt_sinr_s.append(ss)
        print(f"[Mt={Mt}, Nr={Nr_fix}] T_E2E={m:.4f}±{s:.4f} s, SINR={sm:.2f}±{ss:.2f} dB")

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(args.Mt_list, mt_mean, yerr=mt_std, marker='o', capsize=3, label=r"$T_{\sf E2E}$")
    ax.set_xlabel("Transmit antennas $M_t$")
    ax.set_ylabel("Mean end-to-end latency $T_{\\sf E2E}$ (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "latency_vs_Mt.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(args.Mt_list, mt_sinr_m, yerr=mt_sinr_s, marker='s', capsize=3, color='C2', label="SINR (dB)")
    ax.set_xlabel("Transmit antennas $M_t$")
    ax.set_ylabel("Mean SINR (dB)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "sinr_vs_Mt.png"), dpi=300)
    plt.close(fig)

    # ---- 2) Latency/SINR vs Nr (fix Mt = source Mt) ----
    Mt_fix = args.Mt_src
    nr_mean, nr_std, nr_sinr_m, nr_sinr_s = [], [], [], []
    for Nr in args.Nr_list:
        m, s, sm, ss = rollout_eval_MtNr(args.U, Mt_fix, Nr, algo_src, args.Mt_src,
                                         steps=args.steps, seed=321, pmax=args.pmax)
        nr_mean.append(m); nr_std.append(s); nr_sinr_m.append(sm); nr_sinr_s.append(ss)
        print(f"[Mt={Mt_fix}, Nr={Nr}] T_E2E={m:.4f}±{s:.4f} s, SINR={sm:.2f}±{ss:.2f} dB")

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(args.Nr_list, nr_mean, yerr=nr_std, marker='o', capsize=3, color='C1', label=r"$T_{\sf E2E}$")
    ax.set_xlabel("Receive antennas $N_r$")
    ax.set_ylabel("Mean end-to-end latency $T_{\\sf E2E}$ (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "latency_vs_Nr.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.errorbar(args.Nr_list, nr_sinr_m, yerr=nr_sinr_s, marker='s', capsize=3, color='C3', label="SINR (dB)")
    ax.set_xlabel("Receive antennas $N_r$")
    ax.set_ylabel("Mean SINR (dB)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "sinr_vs_Nr.png"), dpi=300)
    plt.close(fig)

    print("[OK] Saved: latency_vs_Mt.png, sinr_vs_Mt.png, latency_vs_Nr.png, sinr_vs_Nr.png")

if __name__ == "__main__":
    main()
