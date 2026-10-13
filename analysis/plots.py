import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import best_val_runs, load_test, split_run
from strategies.kickstart import BANDS

#figures for the report from test_results.csv (evaluate.py) and runs/ (training logs)
#eg python -m analysis.plots --methods fixed uniform evolution curriculum kickstart

#categorical colours in a fixed order (colourblind checked), a method keeps its colour in every figure
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_ORDER = ("fixed", "uniform", "curriculum", "evolution", "kickstart")
#light to dark blue for ordered things like the test phase or the len band
ORDINAL_BLUES = ["#86b6ef", "#2a78d6", "#104281"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

def set_style():
    #thin marks, quiet grid and axes, text in ink not series colours
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
                         "font.size": 8, "axes.titlesize": 8, "legend.frameon": False, "legend.fontsize": 7})

def colour_of(method, methods):
    #known methods always have the same slot
    if method in METHOD_ORDER:
        colour = SERIES_COLOURS[METHOD_ORDER.index(method)]
        #same colour in every figure
        return colour
    #others take the free slots: after the known methods, then slots of known methods that arent in this figure
    free_slots = list(range(len(METHOD_ORDER), len(SERIES_COLOURS)))
    for slot, known_method in enumerate(METHOD_ORDER):
        #known method that isnt in this figure
        if known_method not in methods:
            free_slots.append(slot)
    extra_methods = []
    for other in methods:
        #method without a fixed slot
        if other not in METHOD_ORDER:
            extra_methods.append(other)
    #no cycling, a reused colour would look like another method
    if extra_methods.index(method) >= len(free_slots):
        raise ValueError(f"Out of colours for {method}, plot fewer methods at once!")
    colour = SERIES_COLOURS[free_slots[extra_methods.index(method)]]
    #hex colour of this method
    return colour

def save(fig, out, name):
    #pdf for the report, png to look at
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_final_bars(episodes, picks, methods, out):
    #the required plot: avg +- std over the 10 reps per len for the submitted net of each method, overall avg in the title
    fig, axes = plt.subplots(len(methods), 1, figsize=(6.5, 1.6 * len(methods)), sharex=True, sharey=True, squeeze=False)
    for ax, method in zip(axes[:, 0], methods):
        run = episodes[(episodes["method"] == method) & (episodes["seed"] == picks.loc[method, "seed"])]
        per_length = run.groupby("length")["score"].agg(["mean", "std"])
        bar_positions = np.arange(len(per_length))
        ax.bar(bar_positions, per_length["mean"], 0.7, yerr=per_length["std"], color=colour_of(method, methods), edgecolor=SURFACE, linewidth=0.8, error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK})
        ax.set_title(f"{method} (seed {picks.loc[method, 'seed']}): average score over all pole lengths = {per_length['mean'].mean():.0f}", loc="left")
        ax.set_ylabel("episode length")
        ax.grid(axis="x", visible=False)
    #len labels only on the bottom plot since x is shared
    length_labels = []
    for length in per_length.index:
        length_labels.append(f"{length:.2f}")
    axes[-1, 0].set_xticks(bar_positions, length_labels, rotation=45)
    axes[-1, 0].set_xlabel("pole length")
    fig.tight_layout()
    save(fig, out, "test_bars_final_nets")

def plot_per_length(episodes, methods, out):
    #mean over seeds of the per len score, band = +- std over seeds
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    per_seed = episodes.groupby(["method", "seed", "length"])["score"].mean().reset_index()
    for method in methods:
        curve = per_seed[per_seed["method"] == method].groupby("length")["score"].agg(["mean", "std"]).fillna(0)
        colour = colour_of(method, methods)
        ax.plot(curve.index, curve["mean"], color=colour, label=method)
        ax.fill_between(curve.index, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=colour, alpha=0.15, linewidth=0)
    #phase borders so its clear which phase a score falls in
    for border in (500, 1000):
        ax.axhline(border, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("pole length")
    ax.set_ylabel("test score (steps)")
    ax.set_title("test score per pole length, mean +- std over seeds", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, out, "test_per_length")

def plot_phases(episodes, methods, out):
    #share of test episodes that ended in each phase, pooled over seeds and lens
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(methods) + 0.9))
    shares = episodes.groupby("method")["phase"].value_counts(normalize=True).unstack(fill_value=0).reindex(methods)
    left = np.zeros(len(methods))
    phase_names = {1: "fell in phase 1 (normal)", 2: "fell in phase 2 (wind)", 3: "reached phase 3 (growing force)"}
    #stacked left to right, each phase starts where the last one ended
    for phase in (1, 2, 3):
        share = shares.get(phase, pd.Series(0.0, index=methods)).to_numpy()
        ax.barh(methods, share, left=left, height=0.55, color=ORDINAL_BLUES[phase - 1], edgecolor=SURFACE, linewidth=1.5, label=phase_names[phase])
        left += share
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of test episodes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3)
    fig.tight_layout()
    save(fig, out, "test_phases")

def kickstart_curves(run_dir):
    #l_td per val round = trainer loss - mean lam * l_ks over the same updates, plus agreement per band
    cfg = json.loads((run_dir / "cfg.json").read_text())
    val = pd.read_csv(run_dir / "val.csv").set_index("step")
    ks_log = pd.read_csv(run_dir / "ks.csv")
    ks_log["val_round"] = (np.ceil(ks_log["step"] / cfg["val_every"]) * cfg["val_every"]).astype(int)
    #window means back to sums so windows can be merged weighted by their updates
    summed_keys = ["ks", "lam_ks"]
    for band in BANDS:
        summed_keys.append(f"agree_{band}")
    for key in summed_keys:
        ks_log[f"{key}_sum"] = ks_log[key] * ks_log["n_upd"]
    sums = ks_log.groupby("val_round").sum(numeric_only=True)
    curves = pd.DataFrame(index=sums.index)
    curves["l_ks"] = sums["ks_sum"] / sums["n_upd"]
    curves["lam_l_ks"] = sums["lam_ks_sum"] / sums["n_upd"]
    curves["l_td"] = val["loss"].reindex(curves.index) - curves["lam_l_ks"]
    for band in BANDS:
        curves[f"agree_{band}"] = sums[f"agree_{band}_sum"] / sums["n_upd"]
    #cfg is returned too for the lam = 0 line
    return curves, cfg

def plot_kickstart(method, runs_dir, out):
    #training curves of the kickstarted students, mean over seeds
    all_curves = []
    for run_dir in sorted(Path(runs_dir).glob(f"{method}_s*")):
        #only runs of exactly this method that have a ks log
        if split_run(run_dir.name)[0] == method and (run_dir / "ks.csv").exists():
            curves, cfg = kickstart_curves(run_dir)
            all_curves.append(curves)
    #no kickstart logs so nothing to plot
    if not all_curves:
        return
    mean_curves = pd.concat(all_curves).groupby(level=0).mean()
    fig, (ax_loss, ax_agree) = plt.subplots(1, 2, figsize=(6.5, 2.4))
    ax_loss.plot(mean_curves.index, mean_curves["l_td"], color=SERIES_COLOURS[0], label="L_TD")
    ax_loss.plot(mean_curves.index, mean_curves["lam_l_ks"], color=SERIES_COLOURS[1], label="lambda * L_KS")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("loss (mean per update)")
    ax_loss.legend()
    for band_index, band in enumerate(BANDS):
        ax_agree.plot(mean_curves.index, mean_curves[f"agree_{band}"], color=ORDINAL_BLUES[band_index], label=f"{band} poles")
    ax_agree.set_ylabel("greedy action agreement")
    ax_agree.set_ylim(0, 1)
    ax_agree.legend(loc="lower right")
    #lam reaches 0 here, pure dqn after
    for ax in (ax_loss, ax_agree):
        ax.axvline(cfg["ks_decay"] * cfg["steps"], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("env steps (student)")
    fig.suptitle(f"{method}: losses and agreement with the specialists, mean over {len(all_curves)} seeds", x=0.01, ha="left", fontsize=8)
    fig.tight_layout()
    save(fig, out, f"train_{method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, help="run names without _s<seed>, plotted in this order")
    parser.add_argument("--root", default=".", help="dir with runs/")
    parser.add_argument("--test", nargs="+", default=["test_results.csv"], help="csvs from evaluate.py")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    set_style()
    try:
        episodes = load_test(args.methods, args.test)
        picks = best_val_runs(args.methods, f"{args.root}/runs")
        plot_final_bars(episodes, picks, args.methods, args.out)
        plot_per_length(episodes, args.methods, args.out)
        plot_phases(episodes, args.methods, args.out)
        for method in args.methods:
            #training curves only exist for kickstart runs
            if method.startswith("kickstart"):
                plot_kickstart(method, f"{args.root}/runs", args.out)
        print(f"figures saved in {args.out}")
    except FileNotFoundError as missing_file:
        print(f"Missing file: {missing_file}")
    except KeyError as missing_method:
        print(f"No finished runs for method {missing_method}!")
    except ValueError as bad_results:
        print(bad_results)
