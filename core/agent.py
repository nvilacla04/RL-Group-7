import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_script import QNetwork


class DQNAgent:

    def __init__(self, cfg, obs_dim=4, n_act=2):
        self.cfg, self.n_act = cfg, n_act

        self.q = QNetwork(obs_dim, n_act)
        self.tgt = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

        self.n_upd = 0


    @torch.no_grad()
    def act(self, s, eps, rng):
        #eps greedy
        if rng.random() < eps:
            return int(rng.integers(self.n_act))

        return int(self.q(torch.as_tensor(s).unsqueeze(0)).argmax())


    @torch.no_grad()
    def target(self, b):
        q2 = self.tgt(b['s2'])

        if self.cfg.double:
            #online net picks a', tgt net scores it
            q2 = q2.gather(1, self.q(b['s2']).argmax(1, keepdim=True)).squeeze(1)
        else:
            q2 = q2.max(1).values

        return b['r'] + self.cfg.gamma * (1 - b['d']) * q2


    def update(self, b, aux=None, t=0):
        q_all = self.q(b['s'])
        q = q_all.gather(1, b['a'].unsqueeze(1)).squeeze(1)
        y = self.target(b)

        loss = F.smooth_l1_loss(q, y)

        if aux is not None:
            #hook for extra loss terms, eg distillation
            loss = loss + aux(b, q_all, t)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.clip)
        self.opt.step()

        self.n_upd += 1
        if self.n_upd % self.cfg.tgt_every == 0:
            self.sync()

        return loss.item(), (q.detach() - y).abs().mean().item()


    @torch.no_grad()
    def sync(self):
        #polyak, tau=1 is a hard copy
        for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
            pt.mul_(1 - self.cfg.tau).add_(p, alpha=self.cfg.tau)


    def save(self, path):
        torch.save(self.q.state_dict(), path)
