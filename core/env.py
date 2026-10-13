import gym
import numpy as np


X_MAX = 2.4
TH_MAX = 12 * 2 * np.pi / 360
FORCE = 10.0  #default force_mag

#own val lens: cntrs of 10 equal bins on [0.4, 1.8], sit between the test lens
VAL_LENS = tuple(float(x) for x in np.round(np.linspace(0.47, 1.73, 10), 2))



def make_env():
    #unwrapped so no 500 step limit, dynamics same as test script
    return gym.make('CartPole-v1').unwrapped


def reset(env, length, seed=None):
    #only len is changed, polemass_length stays at default like in the test
    env.length = length
    env.force_mag = FORCE  #undo force changes from the stress val
    s, _ = env.reset(seed=seed)

    return s


def shape(s, w_ang, w_pos):
    #+1 per step minus penalty on pole angle and cart pos
    return 1.0 - w_ang * abs(s[2]) / TH_MAX - w_pos * abs(s[0]) / X_MAX
