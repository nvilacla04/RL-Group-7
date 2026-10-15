# RL Group 7: one DQN for every pole length

Code for Assignment 3 of Introduction to Reinforcement Learning (VU, 2026). We train the standard network from `test_script.py` with deep Q-learning so that one network can balance CartPole on every pole length between 0.4 and 1.8.

Our research question is where extra knowledge should go into DQN to get one network that works on all pole lengths: into the data, the loss, or the choice of training lengths. Each strategy below covers one of these, and they are all compared against the same two baselines.

## Setup
We used Python 3.10 with the course versions of gym (0.26.2) and numpy (1.23.5).
```
conda create -n rl python=3.10
conda activate rl
pip install -r requirements.txt
```
Run all commands from the root of the repo. Everything runs on CPU.

## What's in the repo
| Path | What it is |
|---|---|
| `test_script.py` | the course test script with the standard `QNetwork` (not changed) |
| `core/` | shared DQN code used by every method: config, environment and reward, agent, replay buffer, training loop, validation |
| `strategies/evolution.py` | strategy 1: evolution as a data source |
| `strategies/kickstart.py` | strategy 2: kickstarting from specialists |
| `strategies/curriculum.py` | strategy 3: curriculum on the worst pole lengths |
| `train.py` | trains the baselines and strategies 1 and 3 |
| `train_kickstart.py` | trains strategy 2 (it has two stages, see below) |
| `evaluate.py` | runs the official test on saved networks |
| `analysis/` | statistics and figures for the report |
| `dqn_curriculum.ipynb` | notebook with the pilot runs, baselines and strategy 3 |
| `pilot/` | pilot runs we used to choose the hyperparameters |
| `runs/<run>/` | logs of each training run |
| `weights/` | trained networks (see "Trained networks") |

## Shared setup
To keep the comparison fair, every method uses the same setup. Only the part a strategy is about changes.
- The network is the standard `QNetwork` from the test script, trained with double DQN (Huber loss, Adam, learning rate 2.5e-4, one update every 4 env steps, target network copied every 1000 env steps). Epsilon goes from 1 to 0.02 over the first 20% of training.
- The reward is +1 per step minus a penalty for the pole angle and the cart position.
- There is no 500-step cutoff. Training episodes stop at 1000 steps, but that is not stored as a terminal state.
- The only thing we change in the environment is `env.unwrapped.length`, the same way the test script does it.
- Each run is 400k env steps, and every method is run with seeds 0 to 9.
- Every 10k steps the network is validated on our own 10 pole lengths (0.47 to 1.73, which sit between the test lengths). We keep the checkpoint with the best validation score. The test script is never used for training or for choosing checkpoints.

All hyperparameters are in `core/config.py` (strategy 2 adds its own in `KSCfg` in `strategies/kickstart.py`). Any of them can be changed from the command line with `--set name=value`, and every run saves its full config to `runs/<run>/cfg.json`.

## Training
Runs that already finished are skipped, so a command can be stopped and run again. `--procs` sets how many runs go in parallel.

**Baselines**: trained only on the normal length 0.5, or on a random length between 0.4 and 1.8 every episode. The random-length one is our main baseline.
```
python train.py --method fixed --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train.py --method uniform --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
```

**Strategy 1, evolution as a data source**: a small population of networks is evolved on random lengths. Their episodes go into the same replay buffer as the DQN's, and every few generations a copy of the DQN joins the population. The DQN is the network we submit. The `_frozen` run is a control where the population never evolves.
```
python train.py --method evolution --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train.py --method evolution --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --set ev_frozen=true --tag _frozen
```

**Strategy 2, kickstarting from specialists**: three DQNs are first trained on short (0.40 to 0.87), medium (0.87 to 1.33) and long (1.33 to 1.80) poles. The main network then trains on random lengths with an extra loss that pulls it towards the specialist for each transition's pole length. That extra loss fades out halfway through training. `--stage all` trains the specialists first and then the main network. In the main version the specialists' steps come on top of the 400k. The `_eqbudget` version fits everything into 400k (3 x 100k for the specialists and 100k for the main network).
```
python train_kickstart.py --stage all --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train_kickstart.py --stage all --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --tag _eqbudget --set ks_spec_steps=100000 ks_count_spec=true
```

**Strategy 3, curriculum**: the pole lengths are split into 10 bins, and after every validation round the bins where the network scores worst get sampled more often. Some uniform sampling is mixed in so the easier lengths aren't forgotten.
```
python train.py --method curriculum --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
```
The `_anti` run is a control where the same sampler gives the most weight to the easiest lengths instead.
```
python train.py --method curriculum --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --set cl_prio=anti --tag _anti
```

## Testing and results
These steps only use the final saved networks.

1. Run the official test (`test_pole_length` from `test_script.py`, not changed) on every network: 30 pole lengths with 10 episodes each, with fixed seeds so every network gets the same start states. The output is `test_results.csv`.
```
python evaluate.py weights/fixed_s*.pth weights/uniform_s*.pth weights/evolution_s*.pth weights/evolution_frozen_s*.pth weights/curriculum_s*.pth weights/curriculum_anti_s*.pth weights/kickstart_s*.pth weights/kickstart_eqbudget_s*.pth --procs 14
```
2. Statistics: mean and std over the 10 seeds, Welch t-tests and Mann-Whitney U tests against the uniform baseline, between the strategies and of each strategy against its control (Holm corrected within each of these three groups), the phase of the test where the networks fail, and training cost. The tables go in `results/`. `--copy-final` copies each method's final network to `weights/final/`.
```
python -m analysis.stats --methods fixed uniform evolution evolution_frozen curriculum curriculum_anti kickstart kickstart_eqbudget --strategies evolution curriculum kickstart --controls evolution:evolution_frozen curriculum:curriculum_anti kickstart:kickstart_eqbudget --copy-final
```
3. Figures for the report go in `results/figures/`: the bar plot per method (average and std per pole length, with the overall average in the title), score per pole length, phase of failure, and the training curves for strategy 2.
```
python -m analysis.plots --methods fixed uniform evolution evolution_frozen curriculum curriculum_anti kickstart kickstart_eqbudget
```

## Trained networks
- `weights/final/<method>.pth` are the networks we submit, one per method. For each method this is the seed with the best validation score, so the choice doesn't depend on the test.
- `weights/<run>.pth` is the best checkpoint of every single run, for example `weights/kickstart_s3.pth`.

To run the original test script on one of them, set `trained_nn = 'final/kickstart.pth'` (or another method) in `test_script.py`. It loads networks from `weights/`.

## Notes
- The pilot runs in `pilot/` use seeds 100 and up so they don't overlap with the final runs, and only their validation scores were used.
- In gym 0.26, setting `length` does not update `polemass_length`. We left it like that so training matches the test script.
- All training runs and tests were done on one laptop (Intel Core i7-11800H, 8 cores / 16 threads, 32 GB RAM, Windows 11, torch on CPU with 1 thread per run). Every method was trained one command at a time with `--procs 5` and nothing else running, so the wall clock times in `results/cost.csv` can be compared. One 400k-step run took about 4.5 minutes this way.

