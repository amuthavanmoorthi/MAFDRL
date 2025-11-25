import argparse, os, sys, importlib, importlib.util
import numpy as np

def dyn(cls):
    modref, cname = cls.split(":")
    if modref.endswith(".py"):
        path = os.path.abspath(modref)
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(modref)
    return getattr(mod, cname)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-class", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Env = dyn(args.env_class)
    try:
        env = Env(seed=args.seed)
    except TypeError:
        env = Env()

    try:
        obs = env.reset(seed=args.seed)
    except TypeError:
        obs = env.reset()
    if isinstance(obs, dict) and "obs" in obs:
        obs = obs["obs"]
    U = len(obs) if hasattr(obs, "__len__") else 1
    d_obs = np.asarray(obs[0]).reshape(-1).size if U>0 else None

    d_act = None
    if hasattr(env, "action_space"):
        try:
            as0 = env.action_space[0] if hasattr(env.action_space, "__getitem__") else env.action_space
            if hasattr(as0, "shape") and as0.shape is not None:
                d_act = int(np.prod(as0.shape))
            elif hasattr(as0, "n"):
                d_act = int(as0.n)
        except Exception:
            pass

    print(f"U={U}, obs_dim={d_obs}, act_dim={d_act}")
