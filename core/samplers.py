class Sampler:
    #picks a pole len per train ep, update() gets the val results

    def sample(self, rng):
        raise NotImplementedError

    def update(self, v):
        pass

    def probs(self):
        return {}

    def act(self, agent, s, eps, rng):
        #who plays: the learner by default, strategy 1 swaps in population members
        return agent.act(s, eps, rng)

    def dump(self, out):
        #extra logs of a strategy, written next to eps.csv and val.csv
        pass



class Fixed(Sampler):

    def __init__(self, length):
        self.length = length

    def sample(self, rng):
        return self.length



class Uniform(Sampler):

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def sample(self, rng):
        return rng.uniform(self.lo, self.hi)
