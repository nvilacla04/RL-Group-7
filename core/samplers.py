class Sampler:
    #picks a pole len per train ep, update() gets the val results

    def sample(self, rng):
        raise NotImplementedError

    def update(self, v):
        pass

    def probs(self):
        return {}



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
