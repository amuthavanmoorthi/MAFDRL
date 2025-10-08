# mafdrl/experiments/long_train.py
import os
import numpy as np
import torch as th
from scipy.special import expit
from torch.utils.tensorboard import SummaryWriter

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.buffers import ReplayBuffer
from mafdrl.agents.maddpg import MADDPG
from mafdrl.federated.aggregator import fedavg

"""
Paper-quality defaults (tune as needed):
- fed_rounds: 80
- local_iters: 1000
- batch: 128
- buffer size: 200_000
This script auto-selects device='cuda' when available, else 'cpu'.
"""

def long_train(seed=0, U=3, Mt=2, Nr=4,
               batch=128, local_iters=1000, fed_rounds=10000,
               buffer_size=200_000, logdir="runs/mafdrl_long"):

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()}")

    # federated clients (2 by default; increase if you want)
    num_clients = 2
    envs = [MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed+i) for i in range(num_clients)]

    obs_dim = 4            # [q, ||H||_F, f_loc, prev_rho]
    act_dim = 2 + Mt + 1   # [p, rho, w(Mt), f_loc]
    agents = [MADDPG(U, obs_dim, act_dim, device=device) for _ in range(num_clients)]
    buffers = [ReplayBuffer(buffer_size, obs_dim, act_dim, U) for _ in range(num_clients)]

    writer = SummaryWriter(log_dir=logdir)
    round_rewards = []

    for r in range(fed_rounds):
        print(f"--- Federated Round {r+1}/{fed_rounds} ---")
        local_actor_states = []
        client_round_rews = []

        for c, (env, algo, buf) in enumerate(zip(envs, agents, buffers)):
            obs = env.reset()
            last_info = {}
            batch_data = None
            step_rews = []

            for t in range(local_iters):
                acts = algo.act(obs, noise_scale=0.2)
                # map to valid ranges
                acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)   # p
                acts[:, 1] = expit(acts[:, 1])                            # rho
                # w left as-is; env normalizes
                f_min = 0.10 * env.f_loc_max
                acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

                nobs, rew, done, trunc, info = env.step(acts)
                buf.store(obs, acts, rew, nobs, done)
                obs = nobs
                last_info = info
                step_rews.append(float(np.mean(rew)))

                if buf.ready(batch) and (t % 10 == 0):
                    batch_data = buf.sample(batch)
                    algo.update(batch_data, p_bounds=(0.0, env.p_max))

                if (t + 1) % 200 == 0:
                    mean_rew = float(np.mean(batch_data[2])) if batch_data is not None else float('nan')
                    mean_sinr_db = (
                        float(10 * np.log10(max(np.mean(last_info.get("sinr", [1e-12])), 1e-12)))
                        if last_info else float('nan')
                    )
                    print(f"[client {c}] t={t+1:04d} | mean_rew={mean_rew:.3f} | mean_sinr={mean_sinr_db:.2f} dB")
                    global_step = r * local_iters + (t + 1)
                    if not np.isnan(mean_rew):
                        writer.add_scalar(f"client{c}/mean_rew", mean_rew, global_step)
                    if not np.isnan(mean_sinr_db):
                        writer.add_scalar(f"client{c}/mean_sinr_db", mean_sinr_db, global_step)

            client_mean = float(np.mean(step_rews)) if step_rews else float('nan')
            client_round_rews.append(client_mean)
            local_actor_states.append(algo.get_actor_params())

        # FedAvg (actors)
        avg_state_list = fedavg(local_actor_states)
        for algo in agents:
            algo.set_actor_params(avg_state_list)

        round_mean = float(np.nanmean(client_round_rews))
        round_rewards.append(round_mean)
        print(f"[round {r+1}] mean reward across clients = {round_mean:.3f}")
        writer.add_scalar("train_long/round_mean_reward", round_mean, r + 1)

    # Save checkpoints from client 0
    os.makedirs("checkpoints_long", exist_ok=True)
    first = agents[0]
    saved_paths = []
    for i, sd in enumerate(first.get_actor_params()):
        path = f"checkpoints_long/actor_agent{i}.pt"
        th.save(sd, path)
        saved_paths.append(path)
    print("Saved actor checkpoints: " + ", ".join(saved_paths))

    # Plot round curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(round_rewards) + 1), round_rewards, marker='o')
    plt.xlabel("Federated round"); plt.ylabel("Mean reward")
    # plt.title("Long Training: Reward per Round")
    plt.tight_layout()
    plt.savefig("training_reward_long.png", dpi=150)
    print("Saved plot: training_reward_long.png")

    writer.close()
    print("Long training finished.")

if __name__ == "__main__":
    # Default long-run; adjust if needed
    long_train(fed_rounds=80, local_iters=1000, batch=128, buffer_size=200_000)
