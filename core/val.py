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


@torch.no_grad()
def run_stress(net, env, length, cap, seed=None):
    #same force changes as test_pole_length in the test script but capped
    s = reset(env, length, seed)
    n = 0

    while n < cap:
        a = int(net(torch.as_tensor(s).unsqueeze(0)).argmax())
        s, _, term, _, _ = env.step(a)
        n += 1

        if term:
            break

        if 500 <= n <= 1000 and n % 25 == 0:
            env.force_mag = 75
        if n > 1000:
            env.force_mag = 25 + 0.01 * n

    return n



def validate(net, cfg, seed=10_000):
    #greedy runs on own val lens, nominal dynamics
    #fixed seeds so every ckpt is scored on the same starts
    env = make_env()
    lens = np.asarray(cfg.val_lens)
    steps, rets, stress = np.zeros(len(lens)), np.zeros(len(lens)), np.zeros(len(lens))

    for i, l in enumerate(lens):
        for j in range(cfg.val_eps):
            n, r = run_ep(net, env, l, cfg.val_cap, cfg.w_ang, cfg.w_pos, seed + j)
            steps[i] += n / cfg.val_eps
            rets[i] += r / cfg.val_eps

            #same start but with the force changes, only used for picking a 2nd ckpt
            if cfg.stress_cap:
                stress[i] += run_stress(net, env, l, cfg.stress_cap, seed + j) / cfg.val_eps

    #score in [0, 1], 1 = perfect balance for the whole cap
    #stress in [0, 1], 1 = survived the whole stress cap
    return {'lens': lens, 'steps': steps, 'score': rets / cfg.val_cap,
            'stress_steps': stress, 'stress': stress / max(cfg.stress_cap, 1)}
