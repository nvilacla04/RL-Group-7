import numpy as np
import torch

from .env import make_env, reset, shape


@torch.no_grad()
def run_ep(net, env, length, cap, w_ang, w_pos, seed=None):
    s = reset(env, length, seed)
    ret, n = 0.0, 0

    while n < cap:
        a = int(net(torch.as_tensor(s).unsqueeze(0)).argmax())
        s, _, term, _, _ = env.step(a)

        ret += shape(s, w_ang, w_pos)
        n += 1

        if term:
            break

    return n, ret



def validate(net, cfg, seed=10_000):
    #greedy runs on own val lens, nominal dynamics
    #fixed seeds so every ckpt is scored on the same starts
    env = make_env()
    lens = np.asarray(cfg.val_lens)
    steps, rets = np.zeros(len(lens)), np.zeros(len(lens))

    for i, l in enumerate(lens):
        for j in range(cfg.val_eps):
            n, r = run_ep(net, env, l, cfg.val_cap, cfg.w_ang, cfg.w_pos, seed + j)
            steps[i] += n / cfg.val_eps
            rets[i] += r / cfg.val_eps

    #score in [0, 1], 1 = perfect balance for the whole cap
    return {'lens': lens, 'steps': steps, 'score': rets / cfg.val_cap}
