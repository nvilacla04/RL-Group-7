from dataclasses import asdict, dataclass

from .env import VAL_LENS


@dataclass
class Cfg:
    name: str = 'run'
    seed: int = 0
    steps: int = 400_000        #env steps per run, same for every method
    max_ep: int = 1000          #train ep cap, bootstrapped through so no 500 cutoff
    lo: float = 0.4             #train len range
    hi: float = 1.8

    #reward shaping
    w_ang: float = 0.5
    w_pos: float = 0.5

    #dqn
    gamma: float = 0.99
    lr: float = 2.5e-4          #higher lr diverged in the pilot
    batch: int = 128
    buf: int = 100_000
    learn_start: int = 5_000
    train_freq: int = 4         #env steps per update, like mnih 2015
    tau: float = 1.0            #1.0 = hard copy, <1 = polyak
    tgt_every: int = 250        #updates between tgt syncs, so every 1000 env steps
    double: bool = True
    clip: float = 10.0
    eps0: float = 1.0
    eps1: float = 0.02
    eps_frac: float = 0.2       #frac of steps to decay eps over

    #validation
    val_lens: tuple = VAL_LENS
    val_every: int = 10_000
    val_eps: int = 3
    val_cap: int = 1000

    #curriculum (strategy 3)
    cl_prio: str = 'rank'       #rank, prop or anti (control, easiest bins first)
    cl_temp: float = 1.0
    cl_mix: float = 0.2         #uniform mix vs forgetting

    #evolution (strategy 1)
    ev_pop: int = 8             #population size
    ev_lens: int = 2            #lens per gen, every member plays each once
    ev_elite: int = 1           #best members kept unchanged
    ev_tour: int = 3            #tournament size for picking parents
    ev_sigma: float = 0.02      #mutation noise std
    ev_sync: int = 2            #gens between learner copies into the population
    ev_frozen: bool = False     #control run: population never evolves and gets no learner copies


    def dict(self):
        return asdict(self)


    def set(self, kv):
        #overrides like lr=1e-3, cast to the type of the default
        for s in kv:
            k, v = s.split('=', 1)
            t = type(getattr(self, k))

            if t is bool:
                v = v.lower() in ('1', 'true')
            elif t is tuple:
                v = tuple(float(x) for x in v.split(','))
            else:
                v = t(v)

            setattr(self, k, v)

        return self
