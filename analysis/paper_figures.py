import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import INK, MUTED, SURFACE, colour_of, save, set_style
from analysis.stats import best_val_runs, load_test, seed_scores

#the figures in the paper that analysis.plots doesnt make, same style and colours
#eg python -m analysis.paper_figures
METHODS = ["fixed", "uniform", "evolution", "evolution_frozen", "curriculum", "curriculum_anti", "kickstart", "kickstart_eqbudget"]
#strategy next to its control, baselines first
PAIRS = [("fixed", "uniform"), ("evolution", "evolution_frozen"), ("kickstart", "kickstart_eqbudget"), ("curriculum", "curriculum_anti")]
LABELS = {"fixed": "fixed 0.5", "uniform": "uniform", "evolution": "S1 evolution", "evolution_frozen": "S1 frozen",
          "curriculum": "S3 curriculum", "curriculum_anti": "S3 anti", "kickstart": "S2 kickstart", "kickstart_eqbudget": "S2 equal budget"}
SEEDS = range(10)


def plot_bars_submitted(episodes, picks, out):
    #required plot for the submitted net of each method: avg +- std over the 10 reps per len, overall avg in the title
    #made at one column width so the text stays readable in the paper
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.5}):
        fig, axes = plt.subplots(4, 2, figsize=(3.4, 3.7), sharex=True, sharey=True)

        for ax, method in zip(axes.T.ravel(), [m for pair in PAIRS for m in pair]):
            run = episodes[(episodes["method"] == method) & (episodes["seed"] == picks.loc[method, "seed"])]
            per_length = run.groupby("length")["score"].agg(["mean", "std"])
            ax.bar(np.arange(len(per_length)), per_length["mean"] / 1000, 0.75, yerr=per_length["std"] / 1000,
                   color=colour_of(method, METHODS), linewidth=0, error_kw={"elinewidth": 0.4, "capsize": 0.8, "ecolor": INK})
            ax.set_title(f"{LABELS[method]}, seed {picks.loc[method, 'seed']}: avg {per_length['mean'].mean():.0f}", loc="left", pad=2)
            ax.grid(axis="x", visible=False)

        #len labels on the bottom row only, every 5th so they dont overlap
        ticks = np.arange(0, len(per_length), 5)
        for ax in axes[-1]:
            ax.set_xticks(ticks, [f"{l:.2f}" for l in per_length.index[ticks]])
            ax.set_xlabel("pole length")
        for ax in axes[:, 0]:
            ax.set_ylabel("steps (k)")

        fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
        save(fig, out, "bars_submitted")


def gen_curves(method, runs_dir, grid):
    #gen.csv of every seed put on the same env step grid
    curves = []
    for seed in SEEDS:
        gens = pd.read_csv(Path(runs_dir) / f"{method}_s{seed}" / "gen.csv")
        env_steps = gens["pop_steps"] + gens["dqn_steps"]
        curves.append({
            "fit_max": np.interp(grid, env_steps, gens["fit_max"]),
            "fit_mean": np.interp(grid, env_steps, gens["fit_mean"]),
            "pop_share": np.interp(grid, env_steps, gens["pop_steps"] / env_steps),
        })
    #mean over seeds per curve
    return {key: np.mean([c[key] for c in curves], axis=0) for key in curves[0]}


def plot_s1(runs_dir, out):
    #population fitness, rank of the learner copies and the share of env steps played by the population
    grid = np.linspace(10_000, 400_000, 80)
    evo, frozen = gen_curves("evolution", runs_dir, grid), gen_curves("evolution_frozen", runs_dir, grid)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.1))
    steps_k = grid / 1000

    for curves, method in ((evo, "evolution"), (frozen, "evolution_frozen")):
        colour = colour_of(method, METHODS)
        short = "evolution" if method == "evolution" else "frozen"
        axes[0].plot(steps_k, curves["fit_max"], color=colour, label=f"{short} best")
        axes[0].plot(steps_k, curves["fit_mean"], color=colour, linestyle="--", label=f"{short} mean")
        axes[2].plot(steps_k, curves["pop_share"], color=colour, label=LABELS[method])
    axes[0].set_title("population fitness (shaped return)", loc="left")
    #colours are in the legend of the last panel, here only solid = best and dashed = mean
    styles = [plt.Line2D([], [], color=MUTED, label="best member"), plt.Line2D([], [], color=MUTED, linestyle="--", label="population mean")]
    axes[0].legend(handles=styles, loc="center right", bbox_to_anchor=(1, 0.7))

    #every learner copy as a dot, 1 = best member of its gen
    copies = []
    for seed in SEEDS:
        gens = pd.read_csv(Path(runs_dir) / f"evolution_s{seed}" / "gen.csv")
        gens = gens[gens["copy_rank"] >= 0]
        copies.append(pd.DataFrame({"step": (gens["pop_steps"] + gens["dqn_steps"]) / 1000, "rank": gens["copy_rank"] + 1}))
    copies = pd.concat(copies)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(copies))
    axes[1].scatter(copies["step"], copies["rank"] + jitter, s=4, color=colour_of("evolution", METHODS), alpha=0.35, linewidths=0)
    axes[1].axhline(copies["rank"].mean(), color=INK, linewidth=0.8, linestyle="--")
    axes[1].set_ylim(8.6, 0.4)
    axes[1].set_title(f"rank of learner copies (mean {copies['rank'].mean():.1f} of 8)", loc="left")
    axes[1].set_ylabel("rank in population")

    axes[2].set_ylim(0, 1)
    axes[2].set_title("share of env steps played by the population", loc="left")
    axes[2].legend(loc="center right")
    for ax in axes:
        ax.set_xlabel("env steps (k)")

    fig.tight_layout()
    save(fig, out, "s1_evolution")


def plot_s3(runs_dir, out):
    #sampling prob of each len bin during training, mean over seeds, curriculum next to the anti control
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.9), sharey=True)

    for ax, method in zip(axes, ("curriculum", "curriculum_anti")):
        vals = pd.concat([pd.read_csv(Path(runs_dir) / f"{method}_s{seed}" / "val.csv") for seed in SEEDS])
        cols = [c for c in vals.columns if c.startswith("p_")]
        probs = vals.groupby("step")[cols].mean()
        lens = [float(c[2:]) for c in cols]
        image = ax.pcolormesh(probs.index / 1000, lens, probs.T.values, shading="nearest", cmap="Blues", vmin=0, vmax=0.3)
        ax.set_title(LABELS[method], loc="left")
        ax.set_xlabel("env steps (k)")
        ax.grid(False)

    axes[0].set_ylabel("pole length (bin centre)")
    fig.colorbar(image, ax=axes, label="sampling prob", pad=0.02)
    save(fig, out, "s3_curriculum")


def plot_seed_scores(episodes, out):
    #test score of every seed as a dot, bar = mean over seeds, strategy next to its control
    scores = seed_scores(episodes)
    order = [m for pair in PAIRS for m in pair]
    positions = np.arange(len(order)) + np.repeat(np.arange(len(PAIRS)) * 0.6, 2)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    rng = np.random.default_rng(0)

    for x, method in zip(positions, order):
        values = scores.loc[scores["method"] == method, "score"] / 1000
        colour = colour_of(method, METHODS)
        ax.bar(x, values.mean(), 0.75, color=colour, alpha=0.35, edgecolor=colour, linewidth=0.8)
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(values)), values, s=8, color=colour, linewidths=0, zorder=3)

    ax.set_xticks(positions, [LABELS[m] for m in order], rotation=45, ha="right")
    ax.set_ylabel("test score (k steps)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "seed_scores")


def plot_val_curves(runs_dir, out):
    #val score during training, mean +- std over seeds, main baseline and the 3 strategies
    fig, ax = plt.subplots(figsize=(3.6, 2.4))

    for method in ("uniform", "evolution", "kickstart", "curriculum"):
        vals = pd.concat([pd.read_csv(Path(runs_dir) / f"{method}_s{seed}" / "val.csv")[["step", "score"]] for seed in SEEDS])
        curve = vals.groupby("step")["score"].agg(["mean", "std"])
        colour = colour_of(method, METHODS)
        ax.plot(curve.index / 1000, curve["mean"], color=colour, label=LABELS[method])
        ax.fill_between(curve.index / 1000, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=colour, alpha=0.12, linewidth=0)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("env steps (k)")
    ax.set_ylabel("validation score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "val_curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="dir with runs/")
    parser.add_argument("--test", nargs="+", default=["test_results.csv"], help="csvs from evaluate.py")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    set_style()

    episodes = load_test(METHODS, args.test)
    picks = best_val_runs(METHODS, f"{args.root}/runs")
    plot_bars_submitted(episodes, picks, args.out)
    plot_s1(f"{args.root}/runs", args.out)
    plot_s3(f"{args.root}/runs", args.out)
    plot_seed_scores(episodes, args.out)
    plot_val_curves(f"{args.root}/runs", args.out)
    print(f"figures saved in {args.out}")
