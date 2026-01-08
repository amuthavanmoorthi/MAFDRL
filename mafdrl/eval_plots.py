# mafdrl/eval_plots.py
import os
import argparse
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import expit

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.maddpg import MADDPG

mpl.rcParams["font.family"] = "Times New Roman"


# -----------------------------
# ACTION MAPPING (MUST MATCH TRAINING + evaluate.py)
# -----------------------------
def map_actions(env: MECURLLCEnv, acts: np.ndarray) -> np.ndarray:
    acts = np.asarray(acts, dtype=np.float32).copy()

    p_min = 0.05 * env.p_max
    acts[:, 0] = p_min + expit(acts[:, 0]) * (env.p_max - p_min)
    acts[:, 1] = expit(acts[:, 1])

    f_min = 0.10 * env.f_loc_max
    acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

    return acts


def load_trained_policy(checkpoint_dir: str, U: int, obs_dim: int, act_dim: int):
    algo = MADDPG(U, obs_dim, act_dim, device="cpu")
    state_list = []
    for i in range(U):
        path = os.path.join(checkpoint_dir, f"actor_agent{i}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state_list.append(th.load(path, map_location="cpu"))
    algo.set_actor_params(state_list)
    print(f"[OK] Loaded actors from: {checkpoint_dir}")
    return algo


def run_eval_for_seed(env_kwargs, algo, seed, episodes=200, ep_len=80):
    env = MECURLLCEnv(seed=seed, **env_kwargs)

    lat_ms = []
    sinr_db = []
    rhos = []

    for _ in range(episodes):
        obs = env.reset()
        for _ in range(ep_len):
            raw = algo.act(obs, noise_scale=0.0)
            acts = map_actions(env, raw)

            nobs, rew, done, trunc, info = env.step(acts)

            Te2e = np.asarray(info.get("T_e2e", []), dtype=np.float64).reshape(-1)
            sinr = np.asarray(info.get("sinr", []), dtype=np.float64).reshape(-1)  # linear
            rho = np.asarray(info.get("rho", []), dtype=np.float64).reshape(-1)

            if Te2e.size:
                lat_ms.extend((1e3 * Te2e).tolist())
            if sinr.size:
                sinr_db.extend((10.0 * np.log10(np.maximum(sinr, 1e-20))).tolist())
            if rho.size:
                rhos.append(rho.copy())

            obs = nobs
            if bool(np.any(done)) or bool(np.any(trunc)):
                break

    rhos = np.stack(rhos, axis=0) if len(rhos) else np.zeros((0, env_kwargs["U"]), dtype=np.float64)
    return np.asarray(lat_ms, dtype=np.float64), np.asarray(sinr_db, dtype=np.float64), rhos


def plot_latency_cdf(env_kwargs, algo, out_path, seeds, deadline_ms):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(4.2, 3.2))

    for i, s in enumerate(seeds):
        lat_ms, _, _ = run_eval_for_seed(env_kwargs, algo, seed=s, episodes=200, ep_len=80)
        if lat_ms.size == 0:
            continue

        lat_sorted = np.sort(lat_ms)
        cdf = np.arange(1, lat_sorted.size + 1) / lat_sorted.size
        pct = 100.0 * float(np.mean(lat_ms <= deadline_ms))
        plt.plot(lat_sorted, cdf, label=f"Seed {i} ({pct:.1f}%)")

    plt.axvline(deadline_ms, linestyle="--", linewidth=1.0, label="10 ms deadline")
    plt.xlabel(r"End-to-end latency $T_{\mathrm{E2E}}$ (ms)", fontsize=10)
    plt.ylabel("CDF", fontsize=10)
    plt.tick_params(axis="both", labelsize=9)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved latency CDF: {out_path}")


def plot_sinr_histogram(env_kwargs, algo, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _, sinr_db, _ = run_eval_for_seed(env_kwargs, algo, seed=9000, episodes=250, ep_len=80)
    if sinr_db.size == 0:
        raise RuntimeError("No SINR samples collected.")

    rng = float(np.max(sinr_db) - np.min(sinr_db))
    if rng < 1e-3:
        print("[WARN] SINR histogram collapsed (very small range). Policy may output near-zero power OR mapping/checkpoints mismatch.")

    plt.figure(figsize=(4.2, 3.2))
    plt.hist(sinr_db, bins=35, edgecolor="black", linewidth=0.4)

    plt.xlabel("Post-combining SINR (dB)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tick_params(axis="both", labelsize=9)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved SINR histogram: {out_path}")


def plot_offload_distribution(env_kwargs, algo, out_path):
    """
    Boxplot of rho per UE across all collected steps (shows variability).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _, _, rhos = run_eval_for_seed(env_kwargs, algo, seed=10000, episodes=250, ep_len=80)
    if rhos.shape[0] == 0:
        raise RuntimeError("No offloading samples collected.")

    U = env_kwargs["U"]
    data = [rhos[:, u] for u in range(U)]
    means = [float(np.mean(d)) for d in data]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.boxplot(
        data,
        tick_labels=[f"UE{u+1}" for u in range(U)],
        showfliers=False,
        widths=0.5,
    )
    ax.scatter(np.arange(1, U + 1), means, label="UE mean", zorder=3)

    ax.set_xlabel("User equipment (UE)", fontsize=10)
    ax.set_ylabel(r"Offloading ratio $\rho$", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved offloading distribution: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_long")
    parser.add_argument("--U", type=int, default=3)
    parser.add_argument("--Mt", type=int, default=2)
    parser.add_argument("--Nr", type=int, default=4)
    parser.add_argument("--deadline_ms", type=float, default=10.0)
    args = parser.parse_args()

    # Build a temporary env to infer dims (avoids hardcoding obs_dim/act_dim)
    env_tmp = MECURLLCEnv(U=args.U, Mt=args.Mt, Nr=args.Nr, seed=0)
    obs = env_tmp.reset()
    obs_dim = int(obs.shape[1])
    act_dim = int(2 + args.Mt + 1)

    env_kwargs = dict(U=args.U, Mt=args.Mt, Nr=args.Nr)

    algo = load_trained_policy(args.checkpoint_dir, args.U, obs_dim, act_dim)

    plot_latency_cdf(
        env_kwargs, algo,
        out_path="Results/latency_cdf.png",
        seeds=[6000, 7000, 8000],
        deadline_ms=args.deadline_ms
    )
    plot_sinr_histogram(env_kwargs, algo, out_path="Results/sinr_histogram.png")
    plot_offload_distribution(env_kwargs, algo, out_path="Results/offload_distribution.png")


if __name__ == "__main__":
    main()
