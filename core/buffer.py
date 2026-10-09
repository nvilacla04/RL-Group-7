import numpy as np
import torch


class ReplayBuffer:
    #ring buffer, also stores pole len per transition

    def __init__(self, cap, obs_dim=4, seed=0):

        self.s = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.l = np.zeros(cap, np.float32)

        self.cap, self.ptr, self.n = cap, 0, 0
        self.rng = np.random.default_rng(seed)


    def add(self, s, a, r, s2, d, l):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.l[i] = s, a, r, s2, d, l

        #overwrite oldest once full
        self.ptr = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)


    def sample(self, bs):
        idx = self.rng.integers(0, self.n, bs)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ('s', 'a', 'r', 's2', 'd', 'l')}


    def __len__(self):
        return self.n
