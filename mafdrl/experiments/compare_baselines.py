# mafdrl/experiments/compare_baselines.py
import os, csv, argparse, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.special import expit

import torch as th

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.buffers import ReplayBuffer
from mafdrl.agents.maddpg import MADDPG


# ============== IEEE figure style (Times New Roman; Windows-safe) ==============
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


# =========================== Evaluation utilities ===========================
def rollout_eval(policy_fn, env, steps=500, seed=123):
    """Run one evaluation rollout, returning per-step logs."""
    rng = np.random.RandomState(seed)
    obs = env.reset()
    mean_rewards = []
    sinr_list, Te2e_list, Tloc_list, Ttx_list, TQ_list, Tcpu_list = [], [], [], [], [], []
    for t in range(steps):
        acts = policy_fn(obs)
        # range enforcement (same as training/eval elsewhere)
        acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)   # p
        acts[:, 1] = expit(acts[:, 1])                           # rho
        f_min = 0.10 * env.f_loc_max
        acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)
        obs, rew, done, trunc, info = env.step(acts)
        mean_rewards.append(float(np.mean(rew)))
        sinr_list.append(info["sinr"].copy())
        Te2e_list.append(info["T_e2e"].copy())
        Tloc_list.append(info["T_loc"].copy())
        Ttx_list.append(info["T_tx"].copy())
        TQ_list.append(info["T_Q"].copy())
        Tcpu_list.append(info["T_cpu"].copy())
    logs = {
        "mean_rewards": np.asarray(mean_rewards),
        "sinr": np.vstack(sinr_list),
        "Te2e": np.vstack(Te2e_list),
        "Tloc": np.vstack(Tloc_list),
        "Ttx":  np.vstack(Ttx_list),
        "TQ":   np.vstack(TQ_list),
        "Tcpu": np.vstack(Tcpu_list),
    }
    return logs


def summarize_logs(name, logs):
    """Return a row dict with scalar summaries."""
    sinr_db = 10*np.log10(np.clip(logs["sinr"].flatten(), 1e-12, None))
    return {
        "name": name,
        "reward_mean": float(np.mean(logs["mean_rewards"])),
        "reward_std":  float(np.std (logs["mean_rewards"])),
        "Te2e_mean":   float(np.mean(logs["Te2e"])),
        "Ttx_mean":    float(np.mean(logs["Ttx"])),
        "TQ_mean":     float(np.mean(logs["TQ"])),
        "Tloc_mean":   float(np.mean(logs["Tloc"])),
        "Tcpu_mean":   float(np.mean(logs["Tcpu"])),
        "SINR_dB_mean": float(np.mean(sinr_db)),
        "SINR_dB_std":  float(np.std (sinr_db)),
    }


# =========================== Policies ===========================
def load_mafdrl_policy(U, Mt, device="cpu", ckpt_dir="checkpoints_long"):
    """Load federated actors (MA-FDRL) from checkpoints_long/*.pt and return a policy_fn(obs)->acts."""
    algo = MADDPG(U, obs_dim=4, act_dim=2+Mt+1, device=device)
    # Load actor_agent{0..U-1}.pt
    paths = [os.path.join(ckpt_dir, f"actor_agent{i}.pt") for i in range(U)]
    sds = [th.load(p, map_location=device) for p in paths]
    algo.set_actor_params(sds)
    def policy_fn(obs):
        return algo.act(obs, noise_scale=0.0)
    return policy_fn


def train_maddpg_baseline(U, Mt, Nr, steps=20000, batch=128, buffer_size=200_000, seed=0, device="cpu"):
    """Train centralized MADDPG (non-federated) quickly and return policy_fn."""
    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    obs_dim, act_dim = 4, 2+Mt+1
    algo = MADDPG(U, obs_dim, act_dim, device=device)
    buf  = ReplayBuffer(buffer_size, obs_dim, act_dim, U)
    obs = env.reset()
    for t in range(steps):
        acts = algo.act(obs, noise_scale=0.2)
        acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)
        acts[:, 1] = expit(acts[:, 1])
        f_min = 0.10 * env.f_loc_max
        acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)
        nobs, rew, done, trunc, info = env.step(acts)
        buf.store(obs, acts, rew, nobs, done)
        obs = nobs
        if buf.ready(batch) and (t % 10 == 0):
            algo.update(buf.sample(batch), p_bounds=(0.0, env.p_max))
    def policy_fn(o):
        return algo.act(o, noise_scale=0.0)
    return policy_fn


def heuristic_policy_builder(U, Mt):
    """
    Simple rule: moderate rho, proportional beam weight to channel norm, near-max power if backlog,
    and mid-high local CPU.
    """
    def policy_fn(obs):
        # obs shape: (U, obs_dim) with [q, ||H||_F, f_loc, prev_rho]
        o = np.asarray(obs)
        q = o[:, 0]
        h = o[:, 1]
        prev_rho = o[:, 3]
        p = 0.8 + 0.2 * (q > np.median(q))         # higher power if queue high (normalized later)
        rho = 0.5 * np.ones_like(q) * 0.8 + 0.1 * (prev_rho > 0.5)  # bias to mid-high offload
        # w (Mt) dummy logits proportional to channel norm
        w_logits = np.tile(h[:, None], (1, Mt))
        f_loc_logits = 0.7 * np.ones_like(q)       # mid-high CPU
        acts = np.concatenate([p[:, None], rho[:, None], w_logits, f_loc_logits[:, None]], axis=1)
        return acts
    return policy_fn


# =========================== Main comparison ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=3)
    ap.add_argument("--Mt", type=int, default=2)
    ap.add_argument("--Nr", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--maddpg-train-steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--ckpt-mafdrl", type=str, default="checkpoints_long")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Common eval env (same seed for fairness)
    eval_env = MECURLLCEnv(U=args.U, Mt=args.Mt, Nr=args.Nr, seed=1234)

    # ---- Build the three policies ----
    policy_mafdrl = load_mafdrl_policy(args.U, args.Mt, device="cpu", ckpt_dir=args.ckpt_mafdrl)
    policy_maddpg = train_maddpg_baseline(args.U, args.Mt, args.Nr,
                                          steps=args.maddpg_train_steps, seed=args.seed, device=device)
    policy_heur   = heuristic_policy_builder(args.U, args.Mt)

    # ---- Evaluate each (same env; reset between runs) ----
    def eval_policy(policy_fn, name):
        env = MECURLLCEnv(U=args.U, Mt=args.Mt, Nr=args.Nr, seed=1234)  # same seed per policy
        logs = rollout_eval(policy_fn, env, steps=args.eval_steps, seed=1234)
        row = summarize_logs(name, logs)
        return row, logs

    rows = []
    r_mafdrl, logs_mafdrl = eval_policy(policy_mafdrl, "MA-FDRL (federated)")
    r_maddpg, logs_maddpg = eval_policy(policy_maddpg, "MADDPG (centralized)")
    r_heur,   logs_heur   = eval_policy(policy_heur,   "Heuristic")

    rows.extend([r_mafdrl, r_maddpg, r_heur])

    # ---- Save CSV table ----
    csv_path = os.path.join(args.outdir, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] Saved CSV -> {csv_path}")

    # ---- Bar plot: Reward (↑) and Te2e (↓)
    names = [r["name"] for r in rows]
    reward = [r["reward_mean"] for r in rows]
    Te2e   = [r["Te2e_mean"] for r in rows]

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    xpos = np.arange(len(names))
    ax.bar(xpos-0.18, reward, width=0.36, label="Mean reward")
    ax.bar(xpos+0.18, Te2e,  width=0.36, label="Mean $T_{\\sf E2E}$ (s)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Reward / seconds")
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "baseline_bars_reward_te2e.png"), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved plot -> {os.path.join(args.outdir, 'baseline_bars_reward_te2e.png')}")

    # ---- Line plot: Mean reward vs step (for all three)
    xs = np.arange(1, args.eval_steps+1)
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.plot(xs, logs_mafdrl["mean_rewards"], label="MA-FDRL")
    ax.plot(xs, logs_maddpg["mean_rewards"], label="MADDPG")
    ax.plot(xs, logs_heur["mean_rewards"],   label="Heuristic")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean reward")
    ax.legend(frameon=False)
    ax.margins(x=0, y=0)
    fig.savefig(os.path.join(args.outdir, "baseline_reward_curves.png"), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved plot -> {os.path.join(args.outdir, 'baseline_reward_curves.png')}")

    # ---- Bar plot: communication is N/A for baselines (only MA-FDRL shows FL cost).
    # If you want to include comm cost bars, you can extend your training logger to export MB/round into CSV.


if __name__ == "__main__":
    main()
