# compare_baselines.py (place this at project root, same level as mafdrl/)
import os, csv
import numpy as np
import matplotlib.pyplot as plt

EVAL_CSV = "eval_metrics.csv"                      # from mafdrl.eval_plots
HEUR_CSV = "baseline_heuristic_metrics.csv"        # from mafdrl.baselines.heuristic

def read_eval_csv(path):
    """Return dict with arrays from eval_metrics.csv or baseline_heuristic_metrics.csv"""
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def as_array(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)

def main():
    if not os.path.exists(EVAL_CSV):
        print(f"Missing {EVAL_CSV}. Run: python -m mafdrl.eval_plots")
        return
    if not os.path.exists(HEUR_CSV):
        print(f"Missing {HEUR_CSV}. Run: python -m mafdrl.baselines.heuristic")
        return

    # Load
    ev = read_eval_csv(EVAL_CSV)
    he = read_eval_csv(HEUR_CSV)

    # Align lengths to the shorter run (in case steps differ)
    n = min(len(ev), len(he))
    ev = ev[:n]; he = he[:n]
    steps = np.arange(1, n+1)

    # Pull columns
    ev_reward = as_array(ev, "mean_reward")
    ev_sinr   = as_array(ev, "mean_SINR_dB")
    ev_te2e   = as_array(ev, "mean_Te2e") if "mean_Te2e" in ev[0] else as_array(ev, "mean_Te2E")  # be robust

    he_reward = as_array(he, "mean_reward")
    he_sinr   = as_array(he, "mean_sinr_db")
    he_te2e   = as_array(he, "mean_Te2E") if "mean_Te2E" in he[0] else as_array(he, "mean_Te2e")

    # --------- Plot 1: Reward comparison ----------
    plt.figure()
    plt.plot(steps, ev_reward, label="MA-FDRL")
    plt.plot(steps, he_reward, label="Heuristic", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Mean reward")
    # plt.title("Mean Reward vs Step")
    plt.legend(); plt.tight_layout()
    plt.savefig("cmp_reward.png", dpi=150)
    print("Saved: cmp_reward.png")

    # --------- Plot 2: Latency comparison ----------
    plt.figure()
    plt.plot(steps, ev_te2e, label="MA-FDRL")
    plt.plot(steps, he_te2e, label="Heuristic", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Mean T_E2E (s)")
    # plt.title("Mean End-to-End Latency vs Step")
    plt.legend(); plt.tight_layout()
    plt.savefig("cmp_latency.png", dpi=150)
    print("Saved: cmp_latency.png")

    # --------- Plot 3: SINR comparison ----------
    plt.figure()
    plt.plot(steps, ev_sinr, label="MA-FDRL")
    plt.plot(steps, he_sinr, label="Heuristic", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Mean SINR (dB)")
    # plt.title("Mean SINR vs Step")
    plt.legend(); plt.tight_layout()
    plt.savefig("cmp_sinr.png", dpi=150)
    print("Saved: cmp_sinr.png")

    # --------- Summary CSV ----------
    summary = {
        "method": ["MA-FDRL", "Heuristic"],
        "avg_reward": [np.mean(ev_reward), np.mean(he_reward)],
        "avg_Te2E":   [np.mean(ev_te2e),   np.mean(he_te2e)],
        "avg_SINR_dB":[np.mean(ev_sinr),   np.mean(he_sinr)],
        "p95_Te2E":   [np.percentile(ev_te2e,95), np.percentile(he_te2e,95)],
        "p99_Te2E":   [np.percentile(ev_te2e,99), np.percentile(he_te2e,99)],
    }

    with open("cmp_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary.keys())
        for i in range(2):
            w.writerow([summary[k][i] for k in summary.keys()])
    print("Saved: cmp_summary.csv")

if __name__ == "__main__":
    main()
