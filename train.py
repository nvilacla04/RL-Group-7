import argparse
from multiprocessing import Pool
from pathlib import Path

import torch

from core.config import Cfg
from core.samplers import Fixed, Uniform
from core.trainer import train
from strategies.curriculum import Curriculum

#method -> len sampler
METHODS = {
    'fixed': lambda c: Fixed(0.5),
    'uniform': lambda c: Uniform(c.lo, c.hi),
    'curriculum': lambda c: Curriculum(c.val_lens, c.lo, c.hi, c.cl_prio, c.cl_temp, c.cl_mix),
}


def run_name(method, seed, tag=''):
    return f'{method}{tag}_s{seed}'


def run(job):
    #job = (method, seed, cfg overrides, tag, root dir)
    method, seed, kv, tag, root = job
    torch.set_num_threads(1)
    cfg = Cfg(name=run_name(method, seed, tag), seed=seed).set(kv)
    return train(cfg, METHODS[method](cfg), out=f'{root}/runs', wdir=f'{root}/weights')


def launch(jobs, procs=1):
    #skips runs that already finished, so cells can be rerun
    todo = [j for j in jobs if not (Path(j[4]) / 'runs' / run_name(j[0], j[1], j[3]) / 'final.pth').exists()]
    print(f'{len(jobs) - len(todo)} done, {len(todo)} to run')
    if procs > 1 and len(todo) > 1:
        with Pool(min(procs, len(todo))) as pool:
            for r in pool.imap_unordered(run, todo):
                print(r, flush=True)
    else:
        for j in todo:
            print(run(j), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--set', nargs='*', default=[], help='cfg overrides, eg lr=1e-3 steps=100000')
    p.add_argument('--tag', default='', help='suffix for run name')
    p.add_argument('--root', default='.', help='dir for runs/ and weights/')
    p.add_argument('--procs', type=int, default=1)
    a = p.parse_args()
    launch([(a.method, s, a.set, a.tag, a.root) for s in a.seeds], a.procs)
