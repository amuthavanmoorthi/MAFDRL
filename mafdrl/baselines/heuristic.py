# mafdrl/baselines/heuristic.py
import os, csv, glob
import numpy as np
import matplotlib.pyplot as plt

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv

def heuristic_action(env: MECURLLCEnv, U: int, Mt: int,
                     p_frac=0.8, rho_const=0.5, f_frac=0.5):
    """
    Build a fixed action matrix of shape (U, 2+Mt+1):
      [p, rho, w1...w_Mt, f_loc]
    """
    acts = np.zeros((U, 2 + Mt + 1), dtype=float)

    # power & offloading & f_loc
    p = p_frac * env.p_max
    rho = rho_const
    f_loc = f_frac * env.f_loc_max

    acts[:, 0] = p
    acts[:, 1] = rho
    acts[:, 2:2+Mt] = 0.0
    acts[:, 2] = 1.0        # simple direction w = [1, 0, ..., 0]
    acts[:, 2+Mt] = f_loc
    return acts

def run_baseline(U=3, Mt=2, Nr=4, steps=500, seed=777, outdir="."):
    os.makedirs(outdir, exist_ok=True)

    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)

    # buffers
    mean_rewards = []
    mean_sinr_db = []
    mean_Te2e = []

    all_Tloc, all_Ttx, all_TQ, all_Tcpu = [], [], [], []

    obs = env.reset()
    for t in range(steps):
        acts = heuristic_action(env, U, Mt, p_frac=0.8, rho_const=0.5, f_frac=0.5)
        obs, rew, done, trunc, info = env.step(acts)

        mean_rewards.append(float(np.mean(rew)))
        mean_sinr_db.append(float(10*np.log10(max(1e-12, np.mean(info["sinr"])))))
        mean_Te2e.append(float(np.mean(info["T_e2e"])))

        all_Tloc.append(info["T_loc"].copy())
        all_Ttx.append(info["T_tx"].copy())
        all_TQ.append(info["T_Q"].copy())
        all_Tcpu.append(info["T_cpu"].copy())

    # arrays
    mean_rewards = np.asarray(mean_rewards)
    mean_sinr_db = np.asarray(mean_sinr_db)
    mean_Te2e = np.asarray(mean_Te2e)
    Tloc = np.vstack(all_Tloc)
    Ttx  = np.vstack(all_Ttx)
    TQ   = np.vstack(all_TQ)
    Tcpu = np.vstack(all_Tcpu)

    # CSV export
    csv_path = os.path.join(outdir, "baseline_heuristic_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "mean_sinr_db", "mean_Te2E",
                    "avg_Tloc", "avg_Ttx", "avg_TQ", "avg_Tcpu"])
        for i in range(steps):
            w.writerow([
                i+1,
                f"{mean_rewards[i]:.6f}",
                f"{mean_sinr_db[i]:.4f}",
                f"{mean_Te2e[i]:.6e}",
                f"{np.mean(Tloc[i]):.6e}",
                f"{np.mean(Ttx[i]):.6e}",
                f"{np.mean(TQ[i]):.6e}",
                f"{np.mean(Tcpu[i]):.6e}",
            ])
    print(f"Saved CSV: {csv_path}")

    # Plots
    # 1) Reward curve
    plt.figure()
    plt.plot(np.arange(1, len(mean_rewards)+1), mean_rewards)
    plt.xlabel("Step"); plt.ylabel("Mean reward")
    # plt.title("Heuristic Baseline: Reward vs Step")
    plt.tight_layout()
    p1 = os.path.join(outdir, "baseline_heuristic_reward.png")
    plt.savefig(p1, dpi=150)
    print(f"Saved plot: {p1}")

    # 2) SINR curve
    plt.figure()
    plt.plot(np.arange(1, len(mean_sinr_db)+1), mean_sinr_db)
    plt.xlabel("Step"); plt.ylabel("Mean SINR (dB)")
    # plt.title("Heuristic Baseline: Mean SINR vs Step")
    plt.tight_layout()
    p2 = os.path.join(outdir, "baseline_heuristic_sinr.png")
    plt.savefig(p2, dpi=150)
    print(f"Saved plot: {p2}")

    # 3) E2E latency curve
    plt.figure()
    plt.plot(np.arange(1, len(mean_Te2e)+1), mean_Te2e)
    plt.xlabel("Step"); plt.ylabel("Mean T_E2E (s)")
    # plt.title("Heuristic Baseline: Mean End-to-End Latency vs Step")
    plt.tight_layout()
    p3 = os.path.join(outdir, "baseline_heuristic_latency.png")
    plt.savefig(p3, dpi=150)
    print(f"Saved plot: {p3}")

    print("Heuristic baseline finished.")

if __name__ == "__main__":
    run_baseline(steps=500, outdir=".")
