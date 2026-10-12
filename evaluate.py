import argparse
from multiprocessing import Pool
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import torch

from test_script import QNetwork, test_pole_length


# final test for every method: runs saved nets through the course's test function (not changed)
# same 30 lens x 10 eps as test_script.py, but with fixed seeds so every net gets the same start states
LENS = np.linspace(0.4, 1.8, 30)


def score(job):
    path, n, seed = job
    torch.set_num_threads(1)
    q = QNetwork(4, 2)
    q.load_state_dict(torch.load(path, weights_only=True))
    q.eval()

    rows = []
    with torch.no_grad():
        for l in LENS:
            for j in range(n):
                env = gym.make('CartPole-v1')
                env.unwrapped.length = l
                env.reset(seed=seed + j) # seeds the env, so the reset inside test_pole_length gives the same start for every net
                rows.append({'net': Path(path).stem, 'len': round(l, 2), 'ep': j, 'steps': test_pole_length(env, q)})

    print(f'{Path(path).stem}: avg {np.mean([r["steps"] for r in rows]):.0f}', flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('nets', nargs='+', help='.pth files, eg weights/*.pth')
    p.add_argument('--eps', type=int, default=10, help='eps per len, 10 like the test script')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='test_results.csv')
    p.add_argument('--procs', type=int, default=1)
    a = p.parse_args()

    jobs = [(n, a.eps, a.seed) for n in a.nets]
    if a.procs > 1:
        with Pool(min(a.procs, len(jobs))) as pool:
            dfs = pool.map(score, jobs)
    else:
        dfs = [score(j) for j in jobs]

    # one row per episode, so the plots and stats can be done from this file
    pd.concat(dfs).to_csv(a.out, index=False)
