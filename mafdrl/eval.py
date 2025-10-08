# mafdrl/eval.py
import glob, os, time
import numpy as np
import torch as th
from scipy.special import expit
from torch.utils.tensorboard import SummaryWriter

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.maddpg import MADDPG

def load_actors(algo: MADDPG, ckpt_dir="checkpoints"):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "actor_agent*.pt")))
    if not paths:
        print("No checkpoints found; run training first.")
        return False
    sds = [th.load(p, map_location="cpu") for p in paths]
    algo.set_actor_params(sds)
    print(f"Loaded {len(sds)} actor checkpoints from {ckpt_dir}")
    return True

def eval_once(U=3, Mt=2, Nr=4, steps=200, seed=123, log_dir="runs/mafdrl_eval"):
    env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    obs_dim = 4
    act_dim = 2 + Mt + 1
    algo = MADDPG(U, obs_dim, act_dim, device="cpu")
    _ = load_actors(algo)

    writer = SummaryWriter(log_dir=log_dir)
    run_id = int(time.time())

    obs = env.reset()
    rews = []
    for t in range(steps):
        acts = algo.act(obs, noise_scale=0.0)  # deterministic
        acts[:,0] = np.clip(np.abs(acts[:,0]), 0, env.p_max)           # p
        acts[:,1] = expit(acts[:,1])                                    # rho
        acts[:,-1] = np.clip(np.abs(acts[:,-1]), 0, env.f_loc_max)      # f_loc
        obs, rew, done, trunc, info = env.step(acts)
        rews.append(np.mean(rew))

    mean_rew = float(np.mean(rews))
    print(f"Eval: mean reward over {steps} steps = {mean_rew:.3f}")

    # log to TensorBoard
    writer.add_scalar("eval/mean_reward", mean_rew, run_id)
    writer.close()

if __name__ == "__main__":
    eval_once()
