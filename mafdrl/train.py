# mafdrl/train.py
import os
import numpy as np
import torch as th
from scipy.special import expit  # stable sigmoid

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.buffers import ReplayBuffer
from mafdrl.agents.maddpg import MADDPG
from mafdrl.federated.aggregator import fedavg
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=3, suppress=True)

def train(seed=0, U=3, Mt=2, Nr=4, steps=5000, batch=64, local_iters=500, fed_rounds=5):
    """
    Train MADDPG across two federated clients on the MEC-URLLC env.
    - Prints lightweight metrics every 100 steps per client
    - Saves actors to ./checkpoints at the end
    - Plots mean reward per federated round to training_reward.png
    - Logs metrics to TensorBoard under runs/mafdrl

    Note: obs_dim=4 corresponds to state [queue_bits, ||H||_F, f_loc, prev_rho].
    """
    # two federated clients (simulated)
    envs = [MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed+i) for i in range(2)]
    obs_dim = 4            # [queue_bits, ||H||_F, f_loc, prev_rho]
    act_dim = 2 + Mt + 1   # [p, rho, w(Mt), f_loc]
    agents = [MADDPG(U, obs_dim, act_dim, device="cpu") for _ in range(len(envs))]
    buffers = [ReplayBuffer(50000, obs_dim, act_dim, U) for _ in range(len(envs))]

    round_rewards = []  # mean reward across clients per federated round
    writer = SummaryWriter(log_dir="runs/mafdrl")

    for r in range(fed_rounds):
        print(f"--- Federated Round {r+1}/{fed_rounds} ---")
        local_actor_states = []
        client_round_rews = []

        for c, (env, algo, buf) in enumerate(zip(envs, agents, buffers)):
            obs = env.reset()
            last_info = {}
            batch_data = None
            step_rews = []  # track mean reward per step (across U agents)

            for t in range(local_iters):
                acts = algo.act(obs, noise_scale=0.2)

                # Map raw actions to valid ranges
                acts[:, 0] = np.clip(np.abs(acts[:, 0]), 0, env.p_max)   # p in [0, p_max]
                acts[:, 1] = expit(acts[:, 1])                            # rho in (0,1)
                # w left as-is; env normalizes internally
                # f_loc: keep >=10% of max for stability
                f_min = 0.10 * env.f_loc_max
                acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

                # Environment step
                nobs, rew, done, trunc, info = env.step(acts)
                buf.store(obs, acts, rew, nobs, done)
                obs = nobs
                last_info = info
                step_rews.append(float(np.mean(rew)))

                # Learning update
                if buf.ready(batch) and (t % 10 == 0):
                    batch_data = buf.sample(batch)
                    algo.update(batch_data, p_bounds=(0.0, env.p_max))

                # Lightweight metrics every 100 steps
                if (t + 1) % 100 == 0:
                    mean_rew = float(np.mean(batch_data[2])) if batch_data is not None else float('nan')
                    mean_sinr_db = (
                        float(10 * np.log10(max(np.mean(last_info.get("sinr", [1e-12])), 1e-12)))
                        if last_info else float('nan')
                    )
                    print(f"[client {c}] t={t+1:04d} | mean_rew={mean_rew:.3f} | mean_sinr={mean_sinr_db:.2f} dB")

                    # TensorBoard per-client logs (global step = round*local_iters + step)
                    global_step = r * local_iters + (t + 1)
                    if not np.isnan(mean_rew):
                        writer.add_scalar(f"client{c}/mean_rew", mean_rew, global_step)
                    if not np.isnan(mean_sinr_db):
                        writer.add_scalar(f"client{c}/mean_sinr_db", mean_sinr_db, global_step)

            client_mean = float(np.mean(step_rews)) if step_rews else float('nan')
            client_round_rews.append(client_mean)
            local_actor_states.append(algo.get_actor_params())

        # FedAvg across clients (actors only)
        avg_state_list = fedavg(local_actor_states)
        for algo in agents:
            algo.set_actor_params(avg_state_list)

        # Round-level reward summary
        round_mean = float(np.nanmean(client_round_rews))
        round_rewards.append(round_mean)
        print(f"[round {r+1}] mean reward across clients = {round_mean:.3f}")
        writer.add_scalar("train/round_mean_reward", round_mean, r + 1)

    # Save checkpoints for the first client's actors
    os.makedirs("checkpoints", exist_ok=True)
    first = agents[0]
    saved_paths = []
    for i, sd in enumerate(first.get_actor_params()):
        path = f"checkpoints/actor_agent{i}.pt"
        th.save(sd, path)
        saved_paths.append(path)
    print("Saved actor checkpoints: " + ", ".join(saved_paths))

    # Plot training curve (mean reward per federated round)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(round_rewards) + 1), round_rewards, marker='o')
    plt.xlabel("Federated round")
    plt.ylabel("Mean reward")
    # plt.title("Training reward per round")
    plt.tight_layout()
    plt.savefig("training_reward.png", dpi=150)
    print("Saved plot: training_reward.png")

    writer.close()
    print("Training finished.")
    return agents[0]  # return one set for eval

if __name__ == "__main__":
    # Quick run; change numbers as you like
    train(local_iters=100, fed_rounds=2)
