# RL algorithm
import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from .actor_critic import Actor, Critic


class MADDPG:
    def __init__(self, U, obs_dim, act_dim, lr=1e-3, gamma=0.99, tau=0.05, device="cpu"):
        """
        Stable MADDPG for federated MA-FDRL (Option B):
        - Slightly smaller actor LR
        - Much smaller critic LR
        - Faster target update (larger tau)
        - Lower exploration noise in act()
        These changes are only for stability / smoother convergence.
        """
        self.U = U
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau          # was 0.01, now 0.05 for faster target tracking
        self.device = device

        # per-agent actors
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(U)]
        self.actors_t = [copy.deepcopy(a).to(device) for a in self.actors]

        # === learning rates: make critic slower, actor a bit slower too ===
        actor_lr = lr * 0.5      # was lr; slightly slower, smoother updates
        critic_lr = lr * 0.1     # was lr * 0.5; much slower critic to reduce variance

        # optimizers
        self.opt_actors = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]

        # centralized critic takes concat of all obs and all acts
        self.critic = Critic(in_dim=U * (obs_dim + act_dim)).to(device)
        self.critic_t = copy.deepcopy(self.critic).to(device)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.mse = nn.MSELoss()

    def act(self, obs_np, noise_scale=0.05):
        """
        Forward pass through actors with small Gaussian exploration.
        Default noise_scale was 0.1; reduced to 0.05 for smoother reward curve.
        """
        # obs_np: (U, obs_dim)
        obs = th.tensor(obs_np, dtype=th.float32, device=self.device)
        with th.no_grad():
            acts = []
            for u in range(self.U):
                a_u = self.actors[u](obs[u:u + 1]).cpu().numpy().squeeze()
                acts.append(a_u)
        acts = np.stack(acts, axis=0)
        acts += noise_scale * np.random.randn(*acts.shape)
        acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
        return acts

    def update(self, batch, p_bounds):
        obs, act, rew, nobs, done = batch
        B = obs.shape[0]
        obs_t  = th.tensor(obs,  dtype=th.float32, device=self.device)   # (B,U,obs)
        act_t  = th.tensor(act,  dtype=th.float32, device=self.device)   # (B,U,act)
        rew_t  = th.tensor(rew,  dtype=th.float32, device=self.device)   # (B,U)
        nobs_t = th.tensor(nobs, dtype=th.float32, device=self.device)   # (B,U,obs)
        done_t = th.tensor(done, dtype=th.float32, device=self.device)   # (B,U)

        # ---------- target actions ----------
        with th.no_grad():
            nact_list = []
            for u in range(self.U):
                nact_list.append(self.actors_t[u](nobs_t[:, u, :]))
            nact = th.stack(nact_list, dim=1)  # (B,U,act)

            q_next_in = th.cat(
                [nobs_t.reshape(B, -1), nact.reshape(B, -1)],
                dim=1
            )
            q_next = self.critic_t(q_next_in).squeeze(-1)
            y = (
                rew_t.sum(dim=1)
                + (1.0 - done_t.max(dim=1).values) * self.gamma * q_next
            ).detach()

        # ---------- critic update ----------
        q_in = th.cat([obs_t.reshape(B, -1), act_t.reshape(B, -1)], dim=1)
        q = self.critic(q_in).squeeze(-1)
        critic_loss = self.mse(q, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()

        # gradient clipping for critic
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.opt_critic.step()

        # ---------- actors update (each agent maximizes Q) ----------
        for u in range(self.U):
            a_pred = self.actors[u](obs_t[:, u, :])
            a_all = act_t.clone()
            a_all[:, u, :] = a_pred
            q_in_pi = th.cat(
                [obs_t.reshape(B, -1), a_all.reshape(B, -1)],
                dim=1
            )
            actor_loss = -self.critic(q_in_pi).mean()
            self.opt_actors[u].zero_grad()
            actor_loss.backward()

            # gradient clipping for this actor
            th.nn.utils.clip_grad_norm_(self.actors[u].parameters(), max_norm=0.5)

            self.opt_actors[u].step()

        # ---------- soft target updates ----------
        with th.no_grad():
            for u in range(self.U):
                for p, tp in zip(self.actors[u].parameters(), self.actors_t[u].parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def get_actor_params(self):
        return [a.state_dict() for a in self.actors]

    def set_actor_params(self, state_dicts):
        for a, sd in zip(self.actors, state_dicts):
            a.load_state_dict(sd)
