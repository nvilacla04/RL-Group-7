# RL Group 7: one DQN for every CartPole pole length

## Setup
```
conda create -n rl python=3.10
conda activate rl
pip install -r requirements.txt
```
All commands below are run from the repo root.

## Layout
| Path | What it is |
|---|---|
| `test_script.py` | the official test script and the standard `QNetwork` (unchanged) |
| `core/` | shared DQN code: config, env and reward shaping, agent, replay buffer, trainer, validation |
| `strategies/` | one file per strategy, plugged into the shared trainer |
| `train.py` | baselines (`fixed`, `uniform`), strategy 1 (`evolution`) and strategy 3 (`curriculum`) |
| `train_kickstart.py` | strategy 2 (specialists + kickstarted student) |
| `evaluate.py` | final test: the official `test_pole_length` on saved nets, seeded, one row per episode |
| `analysis/` | stats and figures for all methods |
| `pilot/` | pilot runs used to pick hyperparameters (seeds 100+, validation only) |
| `runs/<name>/` | per run: `cfg.json` (every setting), `eps.csv`, `val.csv`, `final.pth` (+ `ks.csv` for strategy 2) |
| `weights/<name>.pth` | best checkpoint per run, picked on our own validation lengths |

## Strategy 2: kickstarting from specialists
Three specialists are trained on short, medium and long poles (bands `[0.4, 0.87]`, `[0.87, 1.33]`, `[1.33, 1.8]`) and then frozen. The student trains on uniform lengths over `[0.4, 1.8]` with loss `L_TD + λ_t · KL(softmax(Q_teacher/τ) || softmax(Q_student/τ))`, using the specialist of each transition's band as the teacher. λ_t decays linearly to 0 halfway through training. All settings are in `KSCfg` in `strategies/kickstart.py`.
```
# main: student gets the same 400k env steps as the baselines, specialists (3 x 150k) on top
python train_kickstart.py --stage all --seeds 0 1 2 3 4 --procs 6
# equal total budget: specialists 3 x 100k + student 100k = 400k
python train_kickstart.py --stage all --seeds 0 1 2 3 4 --procs 6 --tag _eqbudget --set ks_spec_steps=100000 ks_count_spec=true
```

## Test, stats and figures
The test is only run on the final checkpoints. It is never used for training or for picking checkpoints.
```
# official test_pole_length on every net, one row per episode in test_results.csv
python evaluate.py weights/fixed_s*.pth weights/uniform_s*.pth weights/evolution_s*.pth weights/curriculum_s*.pth weights/kickstart_s*.pth --procs 6
# summary, significance tests (Welch t + Mann-Whitney U, Holm corrected), training cost, final net per method
python -m analysis.stats --methods fixed uniform evolution curriculum kickstart --strategies evolution curriculum kickstart --copy-final
# figures in results/figures/
python -m analysis.plots --methods fixed uniform evolution curriculum kickstart
```
`--copy-final` copies the seed with the best validation score of each method to `weights/final/<method>.pth`. These are the submitted standard networks; in `test_script.py` set `trained_nn = 'final/<method>.pth'`.
