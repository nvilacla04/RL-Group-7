import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .agent import DQNAgent
from .buffer import ReplayBuffer
from .env import make_env, reset, shape
from .val import validate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eps_at(cfg, t):
    #linear decay then flat
    frac = min(1.0, t / (cfg.eps_frac * cfg.steps))
    return cfg.eps0 + frac * (cfg.eps1 - cfg.eps0)



def train(cfg, sampler, aux=None, out='runs', wdir='weights'):
    seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out = Path(out) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    (out / 'cfg.json').write_text(json.dumps(cfg.dict(), indent=1))

    env = make_env()
    agent = DQNAgent(cfg)
    buf = ReplayBuffer(cfg.buf, seed=cfg.seed)

    ep_log, val_log = [], []
    best, best_t, val_steps, t0 = -np.inf, 0, 0, time.time()
    loss_sum, td_sum, n_upd = 0.0, 0.0, 0

    l = sampler.sample(rng)
    s = reset(env, l, seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0

    for t in range(1, cfg.steps + 1):

        eps = eps_at(cfg, t)
        a = sampler.act(agent, s, eps, rng)
        s2, _, term, _, _ = env.step(a)
        r = shape(s2, cfg.w_ang, cfg.w_pos)

        buf.add(s, a, r, s2, term, l)
        s = s2
        ep_ret += r
        ep_len += 1

        if t >= cfg.learn_start and t % cfg.train_freq == 0:
            loss, td = agent.update(buf.sample(cfg.batch), aux, t)
            loss_sum, td_sum, n_upd = loss_sum + loss, td_sum + td, n_upd + 1

        #truncation at max_ep is not stored as done so q still bootstraps
        if term or ep_len >= cfg.max_ep:
            ep_log.append({'step': t, 'len': l, 'ep_len': ep_len, 'ret': ep_ret, 'eps': eps})

            l = sampler.sample(rng)
            s = reset(env, l)
            ep_ret, ep_len = 0.0, 0


        if t % cfg.val_every == 0:
            v = validate(agent.q, cfg)
            val_steps += int(v['steps'].sum() * cfg.val_eps)
            sampler.update(v)
            sc = float(v['score'].mean())

            row = {'step': t, 'time': time.time() - t0, 'val_steps': val_steps, 'score': sc,
                   'loss': loss_sum / max(n_upd, 1), 'td': td_sum / max(n_upd, 1)}
            row.update({f's_{x:.2f}': y for x, y in zip(v['lens'], v['score'])})
            row.update({f'n_{x:.2f}': y for x, y in zip(v['lens'], v['steps'])})
            row.update({f'p_{x:.2f}': y for x, y in sampler.probs().items()})
            val_log.append(row)
            loss_sum, td_sum, n_upd = 0.0, 0.0, 0

            #ckpt selection on own val lens only
            if sc > best:
                best, best_t = sc, t
                agent.save(wdir / f'{cfg.name}.pth')

            pd.DataFrame(ep_log).to_csv(out / 'eps.csv', index=False)
            pd.DataFrame(val_log).to_csv(out / 'val.csv', index=False)
            print(f'[{cfg.name}] t={t} val={sc:.3f} best={best:.3f}@{best_t} '
                  f'min_steps={v["steps"].min():.0f} {time.time() - t0:.0f}s', flush=True)


    agent.save(out / 'final.pth')
    pd.DataFrame(ep_log).to_csv(out / 'eps.csv', index=False)
    pd.DataFrame(val_log).to_csv(out / 'val.csv', index=False)

    return {'name': cfg.name, 'best': best, 'best_t': best_t, 'time': time.time() - t0}
