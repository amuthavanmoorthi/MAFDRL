import numpy as np

class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim, U):
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, U, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, U, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, U), dtype=np.float32)
        self.nobs = np.zeros((size, U, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, U), dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.nobs[self.ptr] = next_obs
        self.done[self.ptr] = done.astype(np.float32)
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.full = True

    def ready(self, batch):
        return (self.ptr if not self.full else self.size) >= batch

    def sample(self, batch):
        maxn = self.size if self.full else self.ptr
        idx = np.random.randint(0, maxn, size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.done[idx])
