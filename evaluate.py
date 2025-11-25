# -*- coding: utf-8 -*-
"""
evaluate.py — writes eval_seed*/eval_metrics.csv with per-UE offloading ratios (rho_u*).

Save as:
  C:\\Gaby\\mafdrl-project\\evaluate.py
"""

import os
import sys
import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
#  Basic import setup
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def dynamic_import(class_path: str):
    """
    Import a class from either:
      1) 'package.module:ClassName'
      2) 'path\\to\\file.py:ClassName'
    """
    if ":" not in class_path:
        raise ValueError(
            f'Invalid class path "{class_path}". Use "module:Class" or "file.py:Class".'
        )
    module_ref, class_name = class_path.split(":", 1)

    if module_ref.endswith(".py"):
        file_path = os.path.abspath(module_ref)
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        try:
            return getattr(mod, class_name)
        except AttributeError:
            raise AttributeError(f'Class "{class_name}" not found in file "{file_path}".')

    mod = importlib.import_module(module_ref)
    try:
        return getattr(mod, class_name)
    except AttributeError:
        raise AttributeError(f'Class "{class_name}" not found in module "{module_ref}".')


def _to_bool_flag(x) -> bool:
    """Return True if any agent is done; handle scalars, lists, tuples, numpy arrays."""
    if isinstance(x, (list, tuple)):
        return any(bool(v) for v in x)
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return bool(_np.any(x))
    except Exception:
        pass
    return bool(x)


def step_env(env, actions):
    """
    Handle both Gym (4-tuple) and Gymnasium (5-tuple) step outputs.
    """
    out = env.step(actions)
    if len(out) == 4:
        # (obs, reward, done, info)
        next_obs, reward, done, info = out
        done_flag = _to_bool_flag(done)
    else:
        # (obs, reward, terminated, truncated, info)
        next_obs, reward, terminated, truncated, info = out
        done_flag = _to_bool_flag(terminated) or _to_bool_flag(truncated)
    return next_obs, reward, done_flag, info


def extract_num_agents_from_obs(obs) -> int:
    if isinstance(obs, dict) and "obs" in obs:
        obs = obs["obs"]
    if hasattr(obs, "__len__"):
        return len(obs)
    return 1


def get_metric(info: Dict[str, Any], key: str, default=np.nan) -> float:
    v = info.get(key, default)
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return float(np.mean(v))
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def eval_once(env, policy, max_steps: int, rho_index: int, seed: int) -> pd.DataFrame:
    try:
        obs = env.reset(seed=seed)
    except TypeError:
        obs = env.reset()

    U = extract_num_agents_from_obs(obs)
    rows: List[Dict[str, Any]] = []

    step = 0
    done = False
    while step < max_steps and not done:
        actions = policy.act(obs, deterministic=True)

        # extract rho
        rhos: List[float] = []
        for a in actions:
            a_vec = np.asarray(a).reshape(-1)
            idx = rho_index if rho_index < a_vec.size else (a_vec.size - 1)
            rho_val = float(np.clip(a_vec[idx], 0.0, 1.0))
            rhos.append(rho_val)

        next_obs, reward, done, info = step_env(env, actions)

        row = {
            "step": step,
            "mean_Te2e": get_metric(info, "mean_Te2e"),
            "mean_Ttx": get_metric(info, "mean_Ttx"),
            "mean_TQ": get_metric(info, "mean_TQ"),
            "mean_Tcpu": get_metric(info, "mean_Tcpu"),
            "mean_Tloc": get_metric(info, "mean_Tloc"),
            "mean_SINR_dB": get_metric(info, "mean_SINR_dB"),
        }
        for u in range(U):
            row[f"rho_u{u}"] = rhos[u]
        rows.append(row)

        obs = next_obs
        step += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
#  Main entry
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="eval_seed0")
    ap.add_argument("--env-class", type=str, required=True,
                    help='Env class "package.module:ClassName" or "path\\file.py:ClassName"')
    ap.add_argument("--policy-class", type=str, required=True,
                    help='Policy class "package.module:ClassName" or "path\\file.py:ClassName"')
    ap.add_argument("--rho-index", type=int, default=2)
    ap.add_argument("--policy-act", type=str, default="act")
    ap.add_argument("--U", type=int, default=None)
    ap.add_argument("--obs-dim", type=int, default=None)
    ap.add_argument("--act-dim", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Build env
    EnvClass = dynamic_import(args.env_class)
    try:
        env = EnvClass(seed=args.seed)
    except TypeError:
        env = EnvClass()

    # Infer dims (if not provided)
    U = args.U
    obs_dim = args.obs_dim
    act_dim = args.act_dim

    if obs_dim is None or act_dim is None:
        try:
            obs = env.reset(seed=args.seed)
        except TypeError:
            obs = env.reset()
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        if obs_dim is None:
            obs_dim = np.asarray(obs[0]).reshape(-1).size
        if act_dim is None:
            act_dim = 4

    # -----------------------------------------------------------------
    #  Build policy
    # -----------------------------------------------------------------
    PolicyClass = dynamic_import(args.policy_class)
    try:
        policy = PolicyClass(U, obs_dim, act_dim)
    except TypeError:
        policy = PolicyClass(U=U, obs_dim=obs_dim, act_dim=act_dim)

    # Try loading weights (torch or .load)
    loaded = False
    if hasattr(policy, "load"):
        try:
            policy.load(args.checkpoint)
            loaded = True
        except Exception:
            pass

    if not loaded:
        try:
            import torch
            sd = torch.load(args.checkpoint, map_location="cpu")

            if hasattr(policy, "load_state_dict") and isinstance(sd, dict):
                try:
                    policy.load_state_dict(sd, strict=False)
                    loaded = True
                except Exception:
                    pass

            if not loaded and isinstance(sd, dict):
                if hasattr(policy, "actor") and "actor" in sd and hasattr(policy.actor, "load_state_dict"):
                    try:
                        policy.actor.load_state_dict(sd["actor"], strict=False)
                        loaded = True
                    except Exception:
                        pass
                if hasattr(policy, "critic") and "critic" in sd and hasattr(policy.critic, "load_state_dict"):
                    try:
                        policy.critic.load_state_dict(sd["critic"], strict=False)
                        loaded = True
                    except Exception:
                        pass
        except Exception:
            pass

    if not loaded:
        print("[WARN] Could not load weights into policy automatically. Proceeding with random weights.")

    # Pick action method
    act_fn = getattr(policy, args.policy_act, None)
    if act_fn is None:
        for cand in ("act", "select_action", "predict"):
            if hasattr(policy, cand):
                act_fn = getattr(policy, cand)
                break
    if act_fn is None:
        raise RuntimeError("No action method found. Try --policy-act select_action (or predict).")

    def _shim_act(obs, deterministic=True):
        return act_fn(obs)
    policy.act = _shim_act

    # -----------------------------------------------------------------
    #  Run evaluation and save
    # -----------------------------------------------------------------
    df = eval_once(env, policy, max_steps=args.steps, rho_index=args.rho_index, seed=args.seed)
    out_csv = os.path.join(args.outdir, "eval_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved metrics with rho columns: {out_csv}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
