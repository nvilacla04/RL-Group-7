import copy # deepcopy = full separate copy of a net, so changing the copy doesn't touch the original

import numpy as np
import torch

from core.env import shape # the shaped reward from the core (1 per step minus penalty for angle and cart position)
from core.samplers import Sampler # base class, the trainer talks to every strategy through this
from test_script import QNetwork # same 4-128-128-2 net as the test script, so any member could be submitted


# strategy 1: evolution as a data source
# basically: a small population of nets plays episodes next to the dqn, all into the same buffer,
# so the dqn learns from more varied data (different nets go to different states).
# the dqn is what we submit, not the population.
#
# how it plugs into the trainer (core/trainer.py):
# - at the start of every episode the trainer calls sample() -> we pick who plays and at which pole len
# - for every action the trainer calls act() -> we let that player pick the action
# - everything else (buffer, dqn updates, val, saving the best net, step budget) is the trainer's normal code,
#   so it's the exact same setup as the baselines. population steps also count to the budget
#
# one generation (gen) = every member plays ev_lens episodes, then the dqn plays the same lens, then evolve()
class ERL(Sampler):

    def __init__(self, cfg):
        self.cfg = cfg # all settings, the ev_* ones are in core/config.py
        self.pop = [QNetwork(4, 2).requires_grad_(False) for _ in range(cfg.ev_pop)] # random nets to start, no grads since they only change by mutation
        self.fit = np.zeros(cfg.ev_pop) # fitness = score of each member this gen (sum of its returns)
        self.queue = [] # list of (who, len) still to play this gen
        self.who = None # current player: a member index, or None = the dqn
        self.ret = 0.0 # return of the current episode so far
        self.gen = 0 # gen counter
        self.agent = None # the dqn, the trainer makes it so we only get it in act()


    # trainer calls this at the start of every episode and uses the len we return
    def sample(self, rng):
        # the previous episode just ended, so add its return to that member's score
        # (skip it if the dqn played, the dqn has no fitness)
        if self.who is not None:
            self.fit[self.who] += self.ret

        if not self.queue: # nothing left to play = gen is over
            if self.gen > 0: # no evolve before the 1st gen, nobody has a score yet
                self.evolve(rng)
            self.gen += 1

            # new random lens for this gen, between 0.4 and 1.8
            # same lens for everyone so scores are fair, new lens every gen so they have to work on all lens
            lens = rng.uniform(self.cfg.lo, self.cfg.hi, self.cfg.ev_lens)

            # play order: member 0 on every len, member 1 on every len, ..., then None (the dqn) on every len
            # [*range(n), None] is just [0, 1, ..., n-1, None]
            self.queue = [(i, l) for i in [*range(self.cfg.ev_pop), None] for l in lens]

        self.who, l = self.queue.pop(0) # take the next (who, len) off the front
        self.ret = 0.0 # fresh episode, reset the return
        return l


    # trainer calls this for every action instead of agent.act()
    @torch.no_grad() # no grads needed to just pick an action, saves time and memory
    def act(self, agent, s, eps, rng):
        self.agent = agent # keep it, evolve() needs it to copy the dqn into the population
        if self.who is None:
            return agent.act(s, eps, rng) # dqn's turn, normal eps greedy like the baselines

        # member's turn: add up its reward for the fitness
        # s here is the state after the last step, so this is the reward of the last step
        # (it's one step behind the real return, but the same for every member so the ranking is fine)
        self.ret += shape(s, self.cfg.w_ang, self.cfg.w_pos)

        # members always play greedy (no eps), exploration comes from mutation instead
        # unsqueeze(0) makes it a batch of 1 because the net expects a batch, argmax = action with the highest q value
        return int(self.pop[self.who](torch.as_tensor(s).unsqueeze(0)).argmax())


    # makes the next gen from the current one, called once every member and the dqn have played
    @torch.no_grad() # we change weights by hand here, not with gradients
    def evolve(self, rng):
        c = self.cfg

        # elitism: the best ev_elite members go to the next gen unchanged, so we never lose the best one
        # argsort(-fit) = member indices from best to worst
        new = [self.pop[i] for i in np.argsort(-self.fit)[:c.ev_elite]]

        # fill the rest of the population with children
        while len(new) < c.ev_pop:
            # tournament selection: pick ev_tour random members (no repeats) and the best of them is the parent
            # so better members get picked more often, but weaker ones still have a chance (keeps it diverse)
            k = rng.choice(c.ev_pop, c.ev_tour, replace=False)
            child = copy.deepcopy(self.pop[k[self.fit[k].argmax()]]) # copy of the parent

            # mutate: add a bit of random noise to every weight, ev_sigma = how big the noise is
            # (no crossover, just mutation, keeps it simple)
            for p in child.parameters():
                p.add_(torch.randn_like(p) * c.ev_sigma)
            new.append(child)

        # every ev_sync gens put a copy of the dqn into the population, so good dqn weights can spread there
        # only once the dqn has started learning (n_upd > 0), before that it's just a random net
        # it goes in the last spot, which is always a child, never an elite
        if self.gen % c.ev_sync == 0 and self.agent.n_upd > 0:
            new[-1] = copy.deepcopy(self.agent.q).requires_grad_(False)

        self.pop, self.fit = new, np.zeros(c.ev_pop) # new gen starts with empty scores
