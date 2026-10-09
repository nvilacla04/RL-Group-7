import numpy as np
from scipy.stats import rankdata

from core.samplers import Sampler


class Curriculum(Sampler):
    #strategy 3: sample more from len bins with the worst val score
    #rank prio like plr (jiang et al 2021) mixed with uniform vs forgetting
    #all bins get rescored every val round so no staleness term needed

    def __init__(self, cntrs, lo, hi, prio='rank', temp=1.0, mix=0.2):
        self.c = np.asarray(cntrs, dtype=float)

        #bin edges halfway between cntrs
        mid = (self.c[1:] + self.c[:-1]) / 2
        self.edges = np.concatenate([[lo], mid, [hi]])

        self.prio, self.temp, self.mix = prio, temp, mix
        self.p = np.full(len(self.c), 1 / len(self.c))  #uniform until 1st val


    def update(self, v):
        assert np.allclose(v['lens'], self.c)
        regret = 1 - v['score']

        if self.prio == 'rank':
            h = (1 / rankdata(-regret)) ** (1 / self.temp)
        else:
            h = np.maximum(regret, 1e-6) ** (1 / self.temp)

        self.p = (1 - self.mix) * h / h.sum() + self.mix / len(h)


    def sample(self, rng):
        i = rng.choice(len(self.p), p=self.p)
        return rng.uniform(self.edges[i], self.edges[i + 1])


    def probs(self):
        return dict(zip(self.c, self.p))
