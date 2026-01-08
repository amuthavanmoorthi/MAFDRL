# # ================================================================
# # evaluate.py — Evaluator for MA-FDRL MADDPG (loads 3 actor ckpts)
# # ================================================================
# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# from scipy.special import expit

# from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
# from mafdrl.agents.maddpg import MADDPG


# # -----------------------------
# # MAP RAW ACTIONS TO VALID ENV
# # -----------------------------
# def map_actions(env, acts: np.ndarray) -> np.ndarray:
#     """
#     MUST match train.py mapping.
#     Key fix: power uses sigmoid + floor to prevent p->0 collapse which causes SINR=-120 dB artifacts.
#     """
#     acts = np.asarray(acts, dtype=np.float64).copy()

#     # p in [p_min, p_max] (avoid p=0 collapse)
#     p_min = 0.05 * env.p_max
#     acts[:, 0] = p_min + expit(acts[:, 0]) * (env.p_max - p_min)

#     # rho in (0,1)
#     acts[:, 1] = expit(acts[:, 1])

#     # w: unchanged (env normalizes)

#     # f_loc in [f_min, f_loc_max]
#     f_min = 0.10 * env.f_loc_max
#     acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

#     return acts


# def _mean_or_nan(x):
#     if x is None:
#         return float("nan")
#     arr = np.array(x, dtype=float).reshape(-1)
#     if arr.size == 0:
#         return float("nan")
#     return float(np.mean(arr))


# def _mean_sinr_db(x):
#     if x is None:
#         return float("nan")
#     arr = np.array(x, dtype=float).reshape(-1)
#     if arr.size == 0:
#         return float("nan")
#     mean_lin = float(np.mean(arr))
#     mean_lin = max(mean_lin, 1e-12)
#     return float(10.0 * np.log10(mean_lin))


# def evaluate(
#     seed: int = 0,
#     steps: int = 200,
#     checkpoint_prefix: str = "checkpoints/actor_agent",
#     U: int = 3,
#     Mt: int = 2,
#     Nr: int = 4,
#     obs_dim: int = 4,
#     act_dim: int = 5,
#     outdir: str = "eval_seed0",
# ):
#     # ----------------------------
#     # 1) Environment (MUST match training)
#     # ----------------------------
#     env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)

#     # ----------------------------
#     # 2) Build MADDPG (actors only used for act())
#     # ----------------------------
#     agent = MADDPG(U, obs_dim, act_dim, device="cpu")

#     # ----------------------------
#     # 3) Load ALL actor checkpoints
#     # ----------------------------
#     actor_paths = [f"{checkpoint_prefix}{i}.pt" for i in range(U)]
#     actor_state_list = []

#     for p in actor_paths:
#         path = Path(p)
#         if not path.exists():
#             raise FileNotFoundError(f"Missing checkpoint: {path}")
#         sd = torch.load(path, map_location="cpu")
#         actor_state_list.append(sd)

#     agent.set_actor_params(actor_state_list)
#     print(f"[OK] Loaded {U} actor checkpoints from prefix: {checkpoint_prefix}")

#     # ----------------------------
#     # 4) Rollout
#     # ----------------------------
#     results = {
#         "step": [],
#         "mean_reward": [],
#         "mean_Te2e": [],
#         "mean_Ttx": [],
#         "mean_TQ": [],
#         "mean_Tcpu": [],
#         "mean_Tloc": [],
#         "mean_SINR_dB": [],
#         "rho_u0": [],
#         "rho_u1": [],
#         "rho_u2": [],
#     }

#     obs = env.reset()

#     for t in range(steps):
#         acts = agent.act(obs, noise_scale=0.0)      # deterministic
#         acts = map_actions(env, acts)

#         nobs, rew, done, trunc, info = env.step(acts)

#         results["step"].append(t)
#         results["mean_reward"].append(float(np.mean(rew)))

#         results["mean_Te2e"].append(_mean_or_nan(info.get("T_e2e", None)))
#         results["mean_Ttx"].append(_mean_or_nan(info.get("T_tx", None)))
#         results["mean_TQ"].append(_mean_or_nan(info.get("T_Q", None)))
#         results["mean_Tcpu"].append(_mean_or_nan(info.get("T_cpu", None)))
#         results["mean_Tloc"].append(_mean_or_nan(info.get("T_loc", None)))

#         results["mean_SINR_dB"].append(_mean_sinr_db(info.get("sinr", None)))

#         rho = info.get("rho", [np.nan] * U)
#         rho_arr = np.array(rho, dtype=float).reshape(-1)
#         for u in range(U):
#             val = rho_arr[u] if u < rho_arr.size else np.nan
#             results[f"rho_u{u}"].append(float(val))

#         obs = nobs

#         if bool(np.any(done)) or bool(np.any(trunc)):
#             obs = env.reset()

#     # ----------------------------
#     # 5) Save
#     # ----------------------------
#     outdir_path = Path(outdir)
#     outdir_path.mkdir(parents=True, exist_ok=True)

#     df = pd.DataFrame(results)
#     csv_path = outdir_path / "eval_metrics.csv"
#     df.to_csv(csv_path, index=False)

#     print(f"[OK] Saved evaluation CSV: {csv_path}")

#     # quick sanity print
#     sinr_db_arr = np.array(results["mean_SINR_dB"], dtype=float)
#     if np.isfinite(sinr_db_arr).any():
#         print(f"[INFO] mean SINR(dB): {np.nanmean(sinr_db_arr):.2f} dB")
#     else:
#         print("[WARN] SINR(dB) is all NaN -> check env/info output.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--steps", type=int, default=200)

#     parser.add_argument("--checkpoint_prefix", type=str, default="checkpoints/actor_agent")

#     parser.add_argument("--U", type=int, default=3)
#     parser.add_argument("--Mt", type=int, default=2)
#     parser.add_argument("--Nr", type=int, default=4)

#     parser.add_argument("--obs_dim", type=int, default=4)
#     parser.add_argument("--act_dim", type=int, default=5)
#     parser.add_argument("--outdir", type=str, default="eval_seed0")

#     args = parser.parse_args()

#     evaluate(
#         seed=args.seed,
#         steps=args.steps,
#         checkpoint_prefix=args.checkpoint_prefix,
#         U=args.U,
#         Mt=args.Mt,
#         Nr=args.Nr,
#         obs_dim=args.obs_dim,
#         act_dim=args.act_dim,
#         outdir=args.outdir,
#     )

# ================================================================
# evaluate.py — Evaluator for MA-FDRL MADDPG (U actors)
# Saves step-wise metrics to CSV for debugging
# ================================================================
# ================================================================
# evaluate.py — Evaluator for MA-FDRL MADDPG (3 actors)
# Fixes:
#   - Action mapping matches training (sigmoid + floors)
#   - Correct SINR dB averaging
#   - Prints clear summary metrics
# ================================================================
# ================================================================
# evaluate.py — Evaluator for MA-FDRL MADDPG (3 actors)
# Fixes:
#   - Action mapping matches training (sigmoid + floors)
#   - Correct SINR dB averaging
#   - Prints clear summary metrics
# ================================================================
# ================================================================
# evaluate.py — Evaluator for MA-FDRL MADDPG (3 actors)
# ================================================================




# ================================================================
# evaluate.py — Evaluator for MA-FDRL MADDPG (U actors)
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
# ACTION MAPPING (MUST MATCH TRAINING)
# -----------------------------
def map_actions(env: MECURLLCEnv, acts: np.ndarray) -> np.ndarray:
    """
    Map raw actor outputs -> valid physical ranges.
    Expected raw acts shape: (U, 2+Mt+1) = [p_raw, rho_raw, w_raw(Mt), f_raw]
    """
    acts = np.asarray(acts, dtype=np.float32).copy()

    # p in [p_min, p_max] to avoid p≈0 collapse
    p_min = 0.05 * env.p_max
    acts[:, 0] = p_min + expit(acts[:, 0]) * (env.p_max - p_min)

    # rho in (0,1)
    acts[:, 1] = expit(acts[:, 1])

    # w left as-is (env normalizes internally if it uses w)
    # f_loc >= 10% f_max
    f_min = 0.10 * env.f_loc_max
    acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

    return acts


def _as_1d(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def _mean_or_nan(x):
    arr = _as_1d(x)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _mean_sinr_db_from_linear(sinr_lin):
    arr = _as_1d(sinr_lin)
    if arr.size == 0:
        return float("nan")
    arr = np.maximum(arr, 1e-20)
    return float(10.0 * np.log10(np.mean(arr)))


# ================================================================
# EVALUATION
# ================================================================
def evaluate(
    seed: int,
    steps: int,
    checkpoint_dir: str,
    outdir: str,
    U: int = 3,
    Mt: int = 2,
    Nr: int = 4,
):
    # 1) Env
    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    obs = env.reset()
    obs_dim = int(obs.shape[1])
    act_dim = int(2 + Mt + 1)

    # 2) Agent
    agent = MADDPG(U, obs_dim, act_dim, device="cpu")

    # 3) Load actors
    ckpt_dir = Path(checkpoint_dir)
    state_list = []
    for i in range(U):
        p = ckpt_dir / f"actor_agent{i}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        state_list.append(torch.load(p, map_location="cpu"))
    agent.set_actor_params(state_list)
    print(f"[OK] Loaded {U} actors from: {checkpoint_dir}")

    # 4) Rollout
    Te2e_ms_list = []
    sinr_db_list = []
    p_list = []
    rho_list = []

    results = {
        "step": [],
        "mean_Te2e_ms": [],
        "mean_Ttx_ms": [],
        "mean_TQ_ms": [],
        "mean_Tcpu_ms": [],
        "mean_Tloc_ms": [],
        "mean_SINR_dB": [],
        "mean_p": [],
        "mean_rho": [],
    }

    # Add per-UE rho columns
    for u in range(U):
        results[f"rho_u{u}"] = []

    for t in range(steps):
        raw = agent.act(obs, noise_scale=0.0)
        acts = map_actions(env, raw)

        nobs, rew, done, trunc, info = env.step(acts)

        # Extract info (expect per-UE arrays)
        Te2e = np.asarray(info.get("T_e2e", []), dtype=np.float64).reshape(-1)
        Ttx  = np.asarray(info.get("T_tx",  []), dtype=np.float64).reshape(-1)
        TQ   = np.asarray(info.get("T_Q",   []), dtype=np.float64).reshape(-1)
        Tcpu = np.asarray(info.get("T_cpu", []), dtype=np.float64).reshape(-1)
        Tloc = np.asarray(info.get("T_loc", []), dtype=np.float64).reshape(-1)

        sinr_lin = np.asarray(info.get("sinr", []), dtype=np.float64).reshape(-1)
        rho = np.asarray(info.get("rho", []), dtype=np.float64).reshape(-1)
        p = np.asarray(info.get("p", acts[:, 0]), dtype=np.float64).reshape(-1)

        # Convert to ms for logging
        Te2e_ms = 1e3 * Te2e
        Ttx_ms  = 1e3 * Ttx
        TQ_ms   = 1e3 * TQ
        Tcpu_ms = 1e3 * Tcpu
        Tloc_ms = 1e3 * Tloc

        # Store scalars
        results["step"].append(t)
        results["mean_Te2e_ms"].append(_mean_or_nan(Te2e_ms))
        results["mean_Ttx_ms"].append(_mean_or_nan(Ttx_ms))
        results["mean_TQ_ms"].append(_mean_or_nan(TQ_ms))
        results["mean_Tcpu_ms"].append(_mean_or_nan(Tcpu_ms))
        results["mean_Tloc_ms"].append(_mean_or_nan(Tloc_ms))
        results["mean_SINR_dB"].append(_mean_sinr_db_from_linear(sinr_lin))
        results["mean_p"].append(_mean_or_nan(p))
        results["mean_rho"].append(_mean_or_nan(rho))

        for u in range(U):
            results[f"rho_u{u}"].append(float(rho[u]) if u < rho.size else float("nan"))

        # For summary stats
        if Te2e_ms.size > 0:
            Te2e_ms_list.append(float(np.mean(Te2e_ms)))
        if sinr_lin.size > 0:
            sinr_db_list.append(float(10.0 * np.log10(np.mean(np.maximum(sinr_lin, 1e-20)))))
        if p.size > 0:
            p_list.append(float(np.mean(p)))
        if rho.size > 0:
            rho_list.append(float(np.mean(rho)))

        obs = nobs
        if bool(np.any(done)) or bool(np.any(trunc)):
            obs = env.reset()

    # 5) Save CSV
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = outdir_path / "eval_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV: {csv_path}")

    # 6) Summary
    deadline_ms = 1e3 * getattr(env, "T_deadline", 0.01)
    Te2e_ms_arr = np.asarray(Te2e_ms_list, dtype=np.float64)
    sinr_db_arr = np.asarray(sinr_db_list, dtype=np.float64)

    success = float(np.mean(Te2e_ms_arr <= deadline_ms)) * 100.0 if Te2e_ms_arr.size else 0.0

    print(f"[Summary] Mean p: {np.mean(p_list):.4f} (p_max={env.p_max})" if p_list else "[Summary] Mean p: NaN")
    print(f"[Summary] Mean rho: {np.mean(rho_list):.3f}" if rho_list else "[Summary] Mean rho: NaN")
    print(f"[Summary] Mean SINR(dB): {np.mean(sinr_db_arr):.2f}" if sinr_db_arr.size else "[Summary] Mean SINR(dB): NaN")
    print(f"[Summary] Mean Te2e: {np.mean(Te2e_ms_arr):.2f} ms" if Te2e_ms_arr.size else "[Summary] Mean Te2e: NaN")
    print(f"[Summary] Percent steps with mean Te2e <= {deadline_ms:.1f} ms: {success:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_long")
    parser.add_argument("--outdir", type=str, default="eval_out")
    parser.add_argument("--U", type=int, default=3)
    parser.add_argument("--Mt", type=int, default=2)
    parser.add_argument("--Nr", type=int, default=4)
    args = parser.parse_args()

    evaluate(
        seed=args.seed,
        steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        outdir=args.outdir,
        U=args.U,
        Mt=args.Mt,
        Nr=args.Nr,
    )
