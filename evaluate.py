# ================================================================
# evaluate.py — Correct evaluator for MA-FDRL MADDPG (3 actors)
# ================================================================
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.maddpg import MADDPG


# -----------------------------
# MAP RAW ACTIONS TO VALID ENV
# -----------------------------
def map_actions(env, acts: np.ndarray) -> np.ndarray:
    """Map raw actor outputs to valid action ranges for MECURLLCEnv."""
    acts = acts.copy()

    # power p in [0, p_max]
    acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)

    # rho in (0, 1)
    acts[:, 1] = expit(acts[:, 1])

    # w — env normalizes internally (no change)

    # f_loc ≥ 10% f_max
    f_min = 0.10 * env.f_loc_max
    acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

    return acts


def _mean_or_nan(x):
    """Return mean of x (array/list/scalar) as float, or NaN if missing/empty."""
    if x is None:
        return float("nan")
    arr = np.array(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _mean_sinr_db(x):
    """Mean SINR in dB from linear SINR array/list/scalar."""
    if x is None:
        return float("nan")
    arr = np.array(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    mean_lin = float(np.mean(arr))
    mean_lin = max(mean_lin, 1e-12)
    return float(10.0 * np.log10(mean_lin))


# ================================================================
# EVALUATION SCRIPT
# ================================================================
def evaluate(
    seed: int = 0,
    steps: int = 200,
    checkpoint_prefix: str = "checkpoints/actor_agent",
    U: int = 3,
    obs_dim: int = 4,
    act_dim: int = 5,
    outdir: str = "eval_seed0",
):
    # ----------------------------
    # 1. Prepare environment
    # ----------------------------
    # Mt, Nr should match training defaults
    env = MECURLLCEnv(U=U, Mt=2, Nr=4, seed=seed)

    # ----------------------------
    # 2. Build MADDPG
    # ----------------------------
    agent = MADDPG(U, obs_dim, act_dim, device="cpu")

    # ----------------------------
    # 3. Load ALL actor checkpoints
    # ----------------------------
    actor_paths = [f"{checkpoint_prefix}{i}.pt" for i in range(U)]

    actor_state_list = []
    for p in actor_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        sd = torch.load(path, map_location="cpu")
        actor_state_list.append(sd)

    agent.set_actor_params(actor_state_list)
    print("[OK] Loaded all 3 actor agents.")

    # ----------------------------
    # 4. Evaluate loop
    # ----------------------------
    results = {
        "step": [],
        "mean_Te2e": [],
        "mean_Ttx": [],
        "mean_TQ": [],
        "mean_Tcpu": [],
        "mean_Tloc": [],
        "mean_SINR_dB": [],
        "rho_u0": [],
        "rho_u1": [],
        "rho_u2": [],
    }

    obs = env.reset()

    for t in range(steps):
        # deterministic evaluation: no exploration noise
        acts = agent.act(obs, noise_scale=0.0)
        acts = map_actions(env, acts)

        nobs, rew, done, trunc, info = env.step(acts)

        results["step"].append(t)

        # latency components are very likely per-UE arrays: average them
        results["mean_Te2e"].append(_mean_or_nan(info.get("T_e2e", None)))
        results["mean_Ttx"].append(_mean_or_nan(info.get("T_tx", None)))
        results["mean_TQ"].append(_mean_or_nan(info.get("T_Q", None)))
        results["mean_Tcpu"].append(_mean_or_nan(info.get("T_cpu", None)))
        results["mean_Tloc"].append(_mean_or_nan(info.get("T_loc", None)))

        # SINR averaged across users, converted to dB
        results["mean_SINR_dB"].append(_mean_sinr_db(info.get("sinr", None)))

        # offloading ratios rho: array of length U
        rho = info.get("rho", [np.nan] * U)
        rho_arr = np.array(rho, dtype=float).reshape(-1)
        for u in range(U):
            val = rho_arr[u] if u < rho_arr.size else np.nan
            results[f"rho_u{u}"].append(float(val))

        obs = nobs

        # safety reset if episode terminates
        if bool(np.any(done)) or bool(np.any(trunc)):
            obs = env.reset()

    # ----------------------------
    # 5. Save CSV
    # ----------------------------
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = outdir_path / "eval_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved evaluation to: {csv_path}")


# ================================================================
# CLI ENTRYPOINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="checkpoints/actor_agent",
    )
    parser.add_argument("--U", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=4)
    parser.add_argument("--act_dim", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="eval_seed0")

    args = parser.parse_args()

    evaluate(
        seed=args.seed,
        steps=args.steps,
        checkpoint_prefix=args.checkpoint_prefix,
        U=args.U,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        outdir=args.outdir,
    )
