# paper-quality
# mafdrl/experiments/long_train.py
import os
import argparse
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

def long_train(
    seed=0, U=3, Mt=2, Nr=4,
    batch=128, local_iters=1000, fed_rounds=80,
    buffer_size=200_000, logroot="runs",
    compress="none", topk_frac=0.01
):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()}")

    num_clients = 2  # number of federated clients
    envs = [MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed + i) for i in range(num_clients)]

    obs_dim = 4            # [q, ||H||_F, f_loc, prev_rho]
    act_dim = 2 + Mt + 1   # [p, rho, w(Mt), f_loc]
    agents = [MADDPG(U, obs_dim, act_dim, device=device) for _ in range(num_clients)]
    buffers = [ReplayBuffer(buffer_size, obs_dim, act_dim, U) for _ in range(num_clients)]

    # ==== set up logging ====
    run_name = f"mafdrl_long_compress-{compress}_seed-{seed}"
    logdir = os.path.join(logroot, run_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    round_rewards = []

    # prepare lightweight args container for fedavg
    class Args: pass
    args = Args()
    args.compress = compress
    args.topk_frac = topk_frac
    args.round_idx = 0

    for r in range(fed_rounds):
        print(f"--- Federated Round {r+1}/{fed_rounds} [{compress}] ---")
        args.round_idx = r + 1

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
                f_min = 0.10 * env.f_loc_max
                acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

                nobs, rew, done, trunc, info = env.step(acts)
                buf.store(obs, acts, rew, nobs, done)
                obs = nobs
                last_info = info
                step_rews.append(float(np.mean(rew)))

                # periodic local update
                if buf.ready(batch) and (t % 10 == 0):
                    batch_data = buf.sample(batch)
                    algo.update(batch_data, p_bounds=(0.0, env.p_max))

                # periodic console logging
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

        # ==== FedAvg aggregation ====
        avg_state_list = fedavg(local_actor_states, args=args, logger=writer)
        for algo in agents:
            algo.set_actor_params(avg_state_list)

        # round summary
        round_mean = float(np.nanmean(client_round_rews))
        round_rewards.append(round_mean)
        print(f"[round {r+1}] mean reward across clients = {round_mean:.3f}")
        writer.add_scalar("train_long/round_mean_reward", round_mean, r + 1)

    # ==== save checkpoints from client 0 ====
    os.makedirs("checkpoints_long", exist_ok=True)
    first = agents[0]
    saved_paths = []
    for i, sd in enumerate(first.get_actor_params()):
        path = f"checkpoints_long/actor_agent{i}.pt"
        th.save(sd, path)
        saved_paths.append(path)
    print("Saved actor checkpoints: " + ", ".join(saved_paths))

    # ==== plot reward curve ====
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(round_rewards) + 1), round_rewards, marker='o')
    plt.xlabel("Federated round")
    plt.ylabel("Mean reward")
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "training_reward_long.png"), dpi=300)
    print(f"Saved plot: {os.path.join(logdir, 'training_reward_long.png')}")

    writer.close()
    print("Long training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fed-rounds", type=int, default=80)
    parser.add_argument("--local-iters", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--compress", type=str, default="none",
                        choices=["none", "topk", "qsgd8", "sign"],
                        help="Communication compression method.")
    parser.add_argument("--topk-frac", type=float, default=0.01,
                        help="Fraction for Top-K sparsification.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    long_train(
        seed=args.seed,
        fed_rounds=args.fed_rounds,
        local_iters=args.local_iters,
        batch=args.batch,
        buffer_size=args.buffer_size,
        compress=args.compress,
        topk_frac=args.topk_frac
    )

