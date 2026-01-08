# # This is the main training file
# # mafdrl/train.py
# import os
# import numpy as np
# import torch as th
# import matplotlib as mpl
# from scipy.special import expit  # stable sigmoid

# from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
# from mafdrl.agents.buffers import ReplayBuffer
# from mafdrl.agents.maddpg import MADDPG
# from mafdrl.federated.aggregator import fedavg
# from torch.utils.tensorboard import SummaryWriter

# mpl.rcParams["font.family"] = "Times New Roman"

# np.set_printoptions(precision=3, suppress=True)


# def _map_actions(env, acts):
#     """
#     Map raw NN outputs to valid physical action ranges.
#     Shared by train() and evaluation.

#     acts: np.ndarray, shape (U, act_dim)
#     """
#     acts = np.asarray(acts, dtype=float).copy()

#     # ---- transmit power p in [p_min, p_max] via sigmoid ----
#     p_min = 0.01 * env.p_max
#     p_max = env.p_max
#     acts[:, 0] = p_min + (p_max - p_min) * expit(acts[:, 0])

#     # ---- rho in (0, 1) via sigmoid ----
#     acts[:, 1] = expit(acts[:, 1])

#     # ---- w: left as-is; env normalizes internally ----

#     # ---- f_loc: keep >= 10% of max for stability ----
#     f_min = 0.10 * env.f_loc_max
#     acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

#     return acts


# def evaluate_policy(U, Mt, Nr, seed, algo, episodes=1, max_steps=500):
#     """
#     Deterministic evaluation (no exploration noise, fixed seed).
#     Returns mean reward per step over evaluation episodes.
#     This is what we will use for the 'clean' Fig. 2 curve.
#     """
#     eval_env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
#     all_ep_means = []

#     for ep in range(episodes):
#         obs = eval_env.reset()
#         step_rews = []

#         for t in range(max_steps):
#             # No exploration noise in evaluation
#             acts = algo.act(obs, noise_scale=0.0)
#             acts = _map_actions(eval_env, acts)

#             nobs, rew, done, trunc, info = eval_env.step(acts)
#             rew = np.asarray(rew, dtype=float)
#             step_rews.append(float(np.mean(rew)))
#             obs = nobs

#             if bool(np.any(done)) or bool(np.any(trunc)):
#                 break

#         all_ep_means.append(float(np.mean(step_rews)) if step_rews else 0.0)

#     return float(np.mean(all_ep_means)) if all_ep_means else 0.0


# def _moving_average(x, window=10):
#     """
#     Simple moving average used to smooth the per-round rewards.
#     """
#     if len(x) < window:
#         return None, None
#     kernel = np.ones(window) / window
#     smoothed = np.convolve(np.array(x, dtype=np.float32), kernel, mode="valid")
#     # x-axis should align with the end of each window
#     rounds = np.arange(window, window + len(smoothed))
#     return rounds, smoothed


# def train(
#     seed=0,
#     U=3,
#     Mt=2,
#     Nr=4,
#     steps=5000,
#     batch=64,
#     local_iters=500,
#     fed_rounds=50,
# ):
#     """
#     Train MADDPG across two federated clients on the MEC-URLLC env.

#     Produces:
#       - training_reward.png      : raw mean round reward (debug)
#       - fig2_smooth.png          : smoothed training curve (for paper)
#       - fig2_eval_ieee.png/.pdf  : deterministic evaluation curve (for paper)
#       - fig3_episode_mean_reward.png/.pdf
#     """
#     # two federated clients (simulated)
#     envs = [MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed + i) for i in range(2)]
#     obs_dim = 4            # [queue_bits, ||H||_F, f_loc, prev_rho]
#     act_dim = 2 + Mt + 1   # [p, rho, w(Mt), f_loc]
#     agents = [MADDPG(U, obs_dim, act_dim, device="cpu") for _ in range(len(envs))]
#     buffers = [ReplayBuffer(50000, obs_dim, act_dim, U) for _ in range(len(envs))]

#     round_rewards = []      # raw mean reward across clients per federated round
#     eval_rewards = []       # deterministic evaluation reward per round
#     writer = SummaryWriter(log_dir="runs/mafdrl")

#     for r in range(fed_rounds):
#         print(f"--- Federated Round {r+1}/{fed_rounds} ---")
#         local_actor_states = []
#         client_round_rews = []

#         for c, (env, algo, buf) in enumerate(zip(envs, agents, buffers)):
#             obs = env.reset()
#             last_info = {}
#             batch_data = None
#             step_rews = []  # track mean reward per step (across U agents)

#             for t in range(local_iters):
#                 # Exploration noise for better learning
#                 acts = algo.act(obs, noise_scale=0.0)
#                 acts = _map_actions(env, acts)

#                 # Environment step
#                 nobs, rew, done, trunc, info = env.step(acts)
#                 rew = np.asarray(rew, dtype=float)

#                 # Guard against NaNs/Infs and clip reward magnitude
#                 rew = np.nan_to_num(rew, nan=-10.0, posinf=10.0, neginf=-10.0)
#                 rew = np.clip(rew, -10.0, 5.0)

#                 buf.store(obs, acts, rew, nobs, done)
#                 obs = nobs
#                 last_info = info
#                 step_rews.append(float(np.mean(rew)))

#                 # Learning update
#                 if buf.ready(batch) and (t % 10 == 0):
#                     batch_data = buf.sample(batch)
#                     algo.update(batch_data, p_bounds=(0.0, env.p_max))

#                 # Lightweight metrics every 100 steps
#                 if (t + 1) % 100 == 0:
#                     mean_rew_log = (
#                         float(np.mean(batch_data[2]))
#                         if batch_data is not None
#                         else float("nan")
#                     )
#                     sinr_vals = np.asarray(last_info.get("sinr", [1e-12]), dtype=float)
#                     sinr_mean_lin = float(np.mean(np.maximum(sinr_vals, 1e-12)))
#                     mean_sinr_db = float(10.0 * np.log10(sinr_mean_lin))

#                     print(
#                         f"[client {c}] t={t+1:04d} | "
#                         f"mean_rew={mean_rew_log:.3f} | "
#                         f"mean_sinr={mean_sinr_db:.2f} dB"
#                     )

#                     # TensorBoard per-client logs (global step = round*local_iters + step)
#                     global_step = r * local_iters + (t + 1)
#                     if not np.isnan(mean_rew_log):
#                         writer.add_scalar(f"client{c}/mean_rew", mean_rew_log, global_step)
#                     writer.add_scalar(f"client{c}/mean_sinr_db", mean_sinr_db, global_step)

#             client_mean = float(np.mean(step_rews)) if step_rews else float("nan")
#             client_round_rews.append(client_mean)
#             local_actor_states.append(algo.get_actor_params())

#         # FedAvg across clients (actors only)
#         avg_state_list = fedavg(local_actor_states)
#         for algo in agents:
#             algo.set_actor_params(avg_state_list)

#         round_mean = float(np.nanmean(client_round_rews))

#         # === NaN guard: detect divergence and stop the outer loop ===
#         if np.isnan(round_mean):
#             print(
#                 f"[round {r+1}] WARNING: round_mean is NaN. "
#                 "Likely divergence in critic/actor. Stopping training."
#             )
#             break  # breaks out of: for r in range(fed_rounds)

#         round_rewards.append(round_mean)
#         print(f"[round {r+1}] mean reward across clients = {round_mean:.3f}")
#         writer.add_scalar("train/round_mean_reward", round_mean, r + 1)

#         # Deterministic evaluation using the first (aggregated) agent
#         eval_seed = seed + 1000
#         eval_rew = evaluate_policy(
#             U, Mt, Nr, eval_seed, agents[0],
#             episodes=1, max_steps=local_iters
#         )
#         eval_rewards.append(eval_rew)
#         print(f"[round {r+1}] eval reward (deterministic) = {eval_rew:.3f}")
#         writer.add_scalar("eval/mean_reward", eval_rew, r + 1)

#     # Save checkpoints for the first client's actors
#     os.makedirs("checkpoints", exist_ok=True)
#     first = agents[0]
#     saved_paths = []
#     for i, sd in enumerate(first.get_actor_params()):
#         path = f"checkpoints/actor_agent{i}.pt"
#         th.save(sd, path)
#         saved_paths.append(path)
#     print("Saved actor checkpoints: " + ", ".join(saved_paths))

#     # ------------------------------------------------------------------
#     # Plot 1: raw training curve (debug)
#     # ------------------------------------------------------------------
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(range(1, len(round_rewards) + 1), round_rewards, marker="o")
#     plt.xlabel("Federated round")
#     plt.ylabel("Mean reward")
#     plt.tight_layout()
#     plt.savefig("training_reward.png", dpi=150)
#     print("Saved plot: training_reward.png")

#     # ------------------------------------------------------------------
#     # Plot 2: smoothed training curve (moving average) - for paper
#     # ------------------------------------------------------------------
#     rounds_s, smooth_rewards = _moving_average(round_rewards, window=10)
#     if smooth_rewards is not None:
#         plt.figure()
#         plt.plot(rounds_s, smooth_rewards, marker="o")
#         plt.xlabel("Federated round")
#         plt.ylabel("Smoothed mean reward")
#         plt.tight_layout()
#         plt.savefig("fig2_smooth.png", dpi=150)
#         print("Saved plot: fig2_smooth.png")

#     # ------------------------------------------------------------------
#     # Plot 3: deterministic evaluation curve - final Fig. 2 (ICC style)
#     # ------------------------------------------------------------------
#     eval_arr = np.array(eval_rewards, dtype=np.float32)

#     # Skip warm-up rounds (policy not stable yet)
#     warmup = 10
#     eval_arr = eval_arr[warmup:]
#     rounds_axis = np.arange(warmup + 1, warmup + 1 + len(eval_arr))

#     # Stronger smoothing for a visually nicer curve (window = 8)
#     if len(eval_arr) >= 8:
#         w = 8
#         kernel = np.ones(w) / w
#         eval_smooth = np.convolve(eval_arr, kernel, mode="valid")
#         rounds_smooth = rounds_axis[w - 1:]
#     else:
#         eval_smooth = eval_arr
#         rounds_smooth = rounds_axis

#     fig, ax = plt.subplots(figsize=(4, 3))  # similar size to other plots

#     # Clean, slightly thicker line; very small markers
#     ax.plot(
#         rounds_smooth,
#         eval_smooth,
#         linewidth=1.5,
#         marker="o",
#         markersize=2.5,
#     )

#     ax.set_xlabel("Federated round", fontsize=10)
#     ax.set_ylabel("Average reward", fontsize=10)
#     ax.tick_params(axis="both", labelsize=9)

#     # Softer grid: only horizontal dashed lines
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

#     # Tight limits to make the curve fill the frame
#     ax.set_xlim(rounds_smooth[0], rounds_smooth[-1])
#     y_min = float(np.min(eval_smooth))
#     y_max = float(np.max(eval_smooth))
#     margin = 0.01 * (y_max - y_min + 1e-8)
#     ax.set_ylim(y_min - margin, y_max + margin)

#     fig.tight_layout()
#     fig.savefig("fig2_eval_ieee.png", dpi=300, bbox_inches="tight")
#     fig.savefig("fig2_eval_ieee.pdf", bbox_inches="tight")
#     print("Saved plots: fig2_eval_ieee.png and fig2_eval_ieee.pdf")

#     # ------------------------------------------------------------------
#     # Fig. 3: Smoothed mean reward per evaluation episode
#     # ------------------------------------------------------------------
#     eval_env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed + 5000)

#     episodes = 100      # many episodes -> good statistics
#     ep_length = 100
#     ep_means = []

#     for ep in range(episodes):
#         obs = eval_env.reset()
#         rewards = []

#         for t in range(ep_length):
#             acts = first.act(obs, noise_scale=0.0)
#             acts = _map_actions(eval_env, acts)

#             nobs, rew, done, trunc, info = eval_env.step(acts)
#             rew = np.asarray(rew, dtype=float)
#             rewards.append(float(np.mean(rew)))
#             obs = nobs

#             if bool(np.any(done)) or bool(np.any(trunc)):
#                 obs = eval_env.reset()

#         ep_means.append(np.mean(rewards))

#     ep_means = np.array(ep_means, dtype=np.float32)

#     # ---- smooth over episodes so curve looks nice ----
#     window = 8                       # you can try 5–10
#     kernel = np.ones(window) / window
#     ep_smooth = np.convolve(ep_means, kernel, mode="valid")
#     ep_axis = np.arange(1, 1 + len(ep_smooth))

#     fig3, ax3 = plt.subplots(figsize=(4, 3))

#     ax3.plot(
#         ep_axis,
#         ep_smooth,
#         linewidth=1.8,
#         marker="o",
#         markersize=2.5,
#     )

#     ax3.set_xlabel("Episode index", fontsize=10)
#     ax3.set_ylabel("Mean reward", fontsize=10)
#     ax3.tick_params(axis="both", labelsize=9)

#     # softer horizontal grid only
#     ax3.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

#     # remove side white gap
#     ax3.set_xlim(ep_axis[0], ep_axis[-1])

#     fig3.tight_layout()
#     fig3.savefig("fig3_episode_mean_reward.png", dpi=300, bbox_inches="tight")
#     fig3.savefig("fig3_episode_mean_reward.pdf", dpi=300, bbox_inches="tight")
#     print("Saved: fig3_episode_mean_reward.png / .pdf")


# if __name__ == "__main__":
#     # Long training for paper-quality learning curve
#     # (Prof. Paul wants ≥1000 steps/round; you set 10k)
#     train(local_iters=10000, fed_rounds=2000)

# This is the main training file
# mafdrl/train.py

import os
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import expit  # stable sigmoid

from mafdrl.envs.mec_urlcc_env import MECURLLCEnv
from mafdrl.agents.buffers import ReplayBuffer
from mafdrl.agents.maddpg import MADDPG
from mafdrl.federated.aggregator import fedavg
from torch.utils.tensorboard import SummaryWriter

mpl.rcParams["font.family"] = "Times New Roman"
np.set_printoptions(precision=3, suppress=True)


def _map_actions(env, acts):
    """
    Map raw NN outputs to valid physical action ranges.
    CRITICAL FIXES:
      - Power uses sigmoid + floor -> prevents p≈0 collapse -> prevents SINR=-120 dB spike
      - rho uses sigmoid
      - f_loc uses sigmoid with floor
      - w is left raw (env normalizes)
    """
    acts = np.asarray(acts, dtype=np.float64)

    # ---- p in [p_min, p_max] (avoid p=0 collapse) ----
    p_min = 0.05 * env.p_max   # 5% power floor (stable SINR)
    acts[:, 0] = p_min + expit(acts[:, 0]) * (env.p_max - p_min)

    # ---- rho in (0,1) ----
    acts[:, 1] = expit(acts[:, 1])

    # ---- f_loc in [f_min, f_loc_max] ----
    f_min = 0.10 * env.f_loc_max
    acts[:, -1] = f_min + expit(acts[:, -1]) * (env.f_loc_max - f_min)

    return acts


def _moving_average(x, window=10):
    if len(x) < window:
        return None, None
    kernel = np.ones(window) / window
    smoothed = np.convolve(np.array(x, dtype=np.float32), kernel, mode="valid")
    rounds = np.arange(window, window + len(smoothed))
    return rounds, smoothed


def evaluate_policy(U, Mt, Nr, seed, algo, episodes=1, max_steps=500):
    """
    Deterministic evaluation:
      - NO exploration noise (noise_scale=0.0)
      - fixed seed
    """
    eval_env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed)
    all_ep_means = []

    for _ in range(episodes):
        obs = eval_env.reset()
        step_rews = []

        for _t in range(max_steps):
            acts = algo.act(obs, noise_scale=0.0)   # FIX: truly deterministic
            acts = _map_actions(eval_env, acts)

            nobs, rew, done, trunc, _info = eval_env.step(acts)
            step_rews.append(float(np.mean(rew)))
            obs = nobs

            if bool(np.any(done)) or bool(np.any(trunc)):
                break

        all_ep_means.append(float(np.mean(step_rews)) if step_rews else 0.0)

    return float(np.mean(all_ep_means)) if all_ep_means else 0.0


def train(seed=0, U=3, Mt=2, Nr=4,
          batch=64, local_iters=500, fed_rounds=50):
    """
    Train MADDPG across two federated clients on MEC-URLLC env.

    Outputs:
      - training_reward.png
      - fig2_smooth.png
      - fig2_eval_ieee.png / .pdf
      - fig3_episode_mean_reward.png / .pdf
      - checkpoints/actor_agent{i}.pt
    """
    envs = [MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed + i) for i in range(2)]
    obs_dim = 4
    act_dim = 2 + Mt + 1

    agents = [MADDPG(U, obs_dim, act_dim, device="cpu") for _ in range(len(envs))]
    buffers = [ReplayBuffer(50000, obs_dim, act_dim, U) for _ in range(len(envs))]

    round_rewards = []
    eval_rewards = []

    writer = SummaryWriter(log_dir="runs/mafdrl")

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
                acts = algo.act(obs, noise_scale=0.05)
                acts = _map_actions(env, acts)

                nobs, rew, done, trunc, info = env.step(acts)
                rew = np.nan_to_num(rew, nan=-1e6, posinf=-1e6, neginf=-1e6)

                buf.store(obs, acts, rew, nobs, done)
                obs = nobs
                last_info = info
                step_rews.append(float(np.mean(rew)))

                if buf.ready(batch) and (t % 10 == 0):
                    batch_data = buf.sample(batch)
                    algo.update(batch_data, p_bounds=(0.0, env.p_max))

                if (t + 1) % 100 == 0:
                    mean_rew = float(np.mean(batch_data[2])) if batch_data is not None else float("nan")

                    sinr_lin = np.asarray(last_info.get("sinr", [0.0]), dtype=np.float64)
                    sinr_mean_lin = float(np.mean(sinr_lin))
                    mean_sinr_db = float(10.0 * np.log10(max(sinr_mean_lin, 1e-12)))

                    print(f"[client {c}] t={t+1:04d} | mean_rew={mean_rew:.3f} | mean_sinr={mean_sinr_db:.2f} dB")

                    global_step = r * local_iters + (t + 1)
                    if not np.isnan(mean_rew):
                        writer.add_scalar(f"client{c}/mean_rew", mean_rew, global_step)
                    writer.add_scalar(f"client{c}/mean_sinr_db", mean_sinr_db, global_step)

            client_mean = float(np.mean(step_rews)) if step_rews else float("nan")
            client_round_rews.append(client_mean)
            local_actor_states.append(algo.get_actor_params())

        # FedAvg across clients (actors only)
        avg_state_list = fedavg(local_actor_states)
        for algo in agents:
            algo.set_actor_params(avg_state_list)

        round_mean = float(np.nanmean(client_round_rews))
        if np.isnan(round_mean):
            print(f"[round {r+1}] WARNING: round_mean is NaN. Stopping training.")
            break

        round_rewards.append(round_mean)
        print(f"[round {r+1}] mean reward across clients = {round_mean:.3f}")
        writer.add_scalar("train/round_mean_reward", round_mean, r + 1)

        # Deterministic evaluation
        eval_seed = seed + 1000
        eval_rew = evaluate_policy(U, Mt, Nr, eval_seed, agents[0], episodes=1, max_steps=local_iters)
        eval_rewards.append(eval_rew)
        print(f"[round {r+1}] eval reward (deterministic) = {eval_rew:.3f}")
        writer.add_scalar("eval/mean_reward", eval_rew, r + 1)

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    first = agents[0]
    saved_paths = []
    for i, sd in enumerate(first.get_actor_params()):
        path = f"checkpoints/actor_agent{i}.pt"
        th.save(sd, path)
        saved_paths.append(path)
    print("Saved actor checkpoints: " + ", ".join(saved_paths))

    # ---------------- Plot 1: raw training curve ----------------
    plt.figure()
    plt.plot(range(1, len(round_rewards) + 1), round_rewards, marker="o")
    plt.xlabel("Federated round")
    plt.ylabel("Mean reward")
    plt.tight_layout()
    plt.savefig("training_reward.png", dpi=150)
    print("Saved plot: training_reward.png")

    # ---------------- Plot 2: smoothed training curve ----------------
    rounds_s, smooth_rewards = _moving_average(round_rewards, window=10)
    if smooth_rewards is not None:
        plt.figure()
        plt.plot(rounds_s, smooth_rewards, marker="o")
        plt.xlabel("Federated round")
        plt.ylabel("Smoothed mean reward")
        plt.tight_layout()
        plt.savefig("fig2_smooth.png", dpi=150)
        print("Saved plot: fig2_smooth.png")

    # ---------------- Plot 3: evaluation curve (IEEE style) ----------------
    eval_arr = np.array(eval_rewards, dtype=np.float32)

    warmup = 10
    eval_arr = eval_arr[warmup:]
    rounds_axis = np.arange(warmup + 1, warmup + 1 + len(eval_arr))

    if len(eval_arr) >= 8:
        w = 8
        kernel = np.ones(w) / w
        eval_smooth = np.convolve(eval_arr, kernel, mode="valid")
        rounds_smooth = rounds_axis[w - 1:]
    else:
        eval_smooth = eval_arr
        rounds_smooth = rounds_axis

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(rounds_smooth, eval_smooth, linewidth=1.5, marker="o", markersize=2.5)
    ax.set_xlabel("Federated round", fontsize=10)
    ax.set_ylabel("Average reward", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlim(rounds_smooth[0], rounds_smooth[-1])
    y_min, y_max = float(np.min(eval_smooth)), float(np.max(eval_smooth))
    margin = 0.01 * (y_max - y_min + 1e-8)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    fig.savefig("fig2_eval_ieee.png", dpi=300, bbox_inches="tight")
    fig.savefig("fig2_eval_ieee.pdf", bbox_inches="tight")
    print("Saved plots: fig2_eval_ieee.png and fig2_eval_ieee.pdf")

    # ---------------- Fig. 3: episode mean reward ----------------
    eval_env = MECURLLCEnv(U=U, Mt=Mt, Nr=Nr, seed=seed + 5000)

    episodes = 100
    ep_length = 100
    ep_means = []

    for _ in range(episodes):
        obs = eval_env.reset()
        rewards = []
        for _t in range(ep_length):
            acts = first.act(obs, noise_scale=0.0)
            acts = _map_actions(eval_env, acts)
            nobs, rew, done, trunc, _info = eval_env.step(acts)
            rewards.append(float(np.mean(rew)))
            obs = nobs
            if bool(np.any(done)) or bool(np.any(trunc)):
                obs = eval_env.reset()
        ep_means.append(float(np.mean(rewards)))

    ep_means = np.array(ep_means, dtype=np.float32)

    window = 8
    kernel = np.ones(window) / window
    ep_smooth = np.convolve(ep_means, kernel, mode="valid")
    ep_axis = np.arange(1, 1 + len(ep_smooth))

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.plot(ep_axis, ep_smooth, linewidth=1.8, marker="o", markersize=2.5)
    ax3.set_xlabel("Episode index", fontsize=10)
    ax3.set_ylabel("Mean reward", fontsize=10)
    ax3.tick_params(axis="both", labelsize=9)
    ax3.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax3.set_xlim(ep_axis[0], ep_axis[-1])

    fig3.tight_layout()
    fig3.savefig("fig3_episode_mean_reward.png", dpi=300, bbox_inches="tight")
    fig3.savefig("fig3_episode_mean_reward.pdf", dpi=300, bbox_inches="tight")
    print("Saved: fig3_episode_mean_reward.png / .pdf")


if __name__ == "__main__":
    # Your paper-quality setting
    train(local_iters=10000, fed_rounds=2000)
