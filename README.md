# RL Group 7: one DQN for every pole length

Code for Assignment 3 of Introduction to Reinforcement Learning (VU, 2026). We train the standard network from `test_script.py` with deep Q-learning so that one network balances CartPole on every pole length between 0.4 and 1.8. Our research question is where extra knowledge should go into DQN: into the data (strategy 1), the loss (strategy 2) or the choice of training lengths (strategy 3). All three are compared against the same baselines, and each has its own control.

## Setup
Python 3.10 with the course versions of gym (0.26.2) and numpy (1.23.5). Run all commands from the repo root; everything runs on CPU.
```
conda create -n rl python=3.10
conda activate rl
pip install -r requirements.txt
```

## What's in the repo
| Path | What it is |
|---|---|
| `test_script.py` | the course test script with the standard `QNetwork` (not changed) |
| `core/` | shared DQN code used by every method: config, environment and reward, agent, replay buffer, training loop, validation |
| `strategies/` | `evolution.py` (strategy 1), `kickstart.py` (strategy 2), `curriculum.py` (strategy 3) |
| `train.py` | trains the baselines and strategies 1 and 3 |
| `train_kickstart.py` | trains strategy 2 (specialists first, then the main network) |
| `evaluate.py` | runs the official test on saved networks |
| `analysis/` | statistics and figures for the report |
| `dqn_curriculum.ipynb` | notebook with the pilot runs, baselines and strategy 3 |
| `pilot/` | pilot runs used to choose the hyperparameters |
| `runs/<run>/` | logs of each training run |
| `weights/` | trained networks |

## Shared setup
Every method uses the same double DQN setup, so only the strategy changes. The only change to the environment is `env.unwrapped.length`, set the same way as in the test script. Each run is 400k env steps with seeds 0 to 9, and there is no 500-step cutoff. Every 10k steps the network is validated on our own 10 pole lengths (between the test lengths), and the best checkpoint is kept. The test script is never used for training or for choosing checkpoints.

All hyperparameters are in `core/config.py` (strategy 2 adds its own in `KSCfg` in `strategies/kickstart.py`). Any of them can be changed with `--set name=value`, and every run saves its full config to `runs/<run>/cfg.json`.

## Training
Finished runs are skipped, so a command can be stopped and rerun. `--procs` sets how many runs go in parallel.

**Baselines**: a fixed length of 0.5, and a random length between 0.4 and 1.8 every episode (the main baseline).
```
python train.py --method fixed --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train.py --method uniform --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
```

**Strategy 1, evolution as a data source**: an evolving population fills the DQN's replay buffer. `_frozen` is the control where the population never evolves.
```
python train.py --method evolution --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train.py --method evolution --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --set ev_frozen=true --tag _frozen
```

**Strategy 2, kickstarting from specialists**: specialists on short, medium and long poles, then a main network pulled towards them by an extra loss. `_eqbudget` is the control that fits everything into 400k steps.
```
python train_kickstart.py --stage all --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train_kickstart.py --stage all --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --tag _eqbudget --set ks_spec_steps=100000 ks_count_spec=true
```

**Strategy 3, curriculum**: the pole lengths where the network scores worst on validation get sampled more often. `_anti` is the control that favours the easiest lengths instead.
```
python train.py --method curriculum --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5
python train.py --method curriculum --seeds 0 1 2 3 4 5 6 7 8 9 --procs 5 --set cl_prio=anti --tag _anti
```

## Testing and results
1. The official test (`test_pole_length`, not changed) on every network: 30 pole lengths, 10 episodes each, with fixed seeds. Output is `test_results.csv`.
```
python evaluate.py weights/fixed_s*.pth weights/uniform_s*.pth weights/evolution_s*.pth weights/evolution_frozen_s*.pth weights/curriculum_s*.pth weights/curriculum_anti_s*.pth weights/kickstart_s*.pth weights/kickstart_eqbudget_s*.pth --procs 14
```
2. Statistics over the 10 seeds (Welch and Mann-Whitney tests with Holm correction, phase of failure, training cost) go in `results/`. `--copy-final` puts each method's submitted network in `weights/final/`.
```
python -m analysis.stats --methods fixed uniform evolution evolution_frozen curriculum curriculum_anti kickstart kickstart_eqbudget --strategies evolution curriculum kickstart --controls evolution:evolution_frozen curriculum:curriculum_anti kickstart:kickstart_eqbudget --copy-final
```
3. Figures go in `results/figures/`. The first command makes the general plots, the second the remaining figures in the paper.
```
python -m analysis.plots --methods fixed uniform evolution evolution_frozen curriculum curriculum_anti kickstart kickstart_eqbudget
python -m analysis.paper_figures
```

## Trained networks
- `weights/final/<method>.pth` are the submitted networks, one per method: the seed with the best validation score, so the choice doesn't depend on the test.
- `weights/<run>.pth` is the best checkpoint of every run, for example `weights/kickstart_s3.pth`.

To run the original test script on one, set `trained_nn = 'final/kickstart.pth'` (or another method) in `test_script.py`.

## Notes
- The pilot runs use seeds 100 and up, and only their validation scores were used.
- In gym 0.26, setting `length` does not update `polemass_length`. We left it like that so training matches the test script.
- Everything was trained and tested on one laptop (Intel Core i7-11800H, 32 GB RAM, Windows 11, 1 CPU thread per run), one command at a time with `--procs 5`, so the wall clock times in `results/cost.csv` are comparable. A 400k-step run takes about 4.5 minutes.
