import argparse
import json
import re
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from strategies.kickstart import BANDS

#stats over seeds for every method: test score summary, significance tests, training cost and the final net pick
#a seed is the unit, its score is the mean test score over the 30 lens x 10 reps
#eg python -m analysis.stats --methods fixed uniform evolution curriculum kickstart --strategies evolution curriculum kickstart
def split_run(name):
    #kickstart_s3 -> (kickstart, 3), stress ckpts kickstart_s3_stress -> (kickstart_stress, 3)
    match = re.match(r"(.+)_s(\d+)(_stress)?$", name)
    if match is None:
        raise ValueError(f"Run name {name} doesnt end in _s<seed>!")
    method_and_seed = (match.group(1) + (match.group(3) or ""), int(match.group(2)))
    #method and seed as a pair
    return method_and_seed

def phase_of(score):
    #fell in phase 1, normal dynamics (0-500)
    if score <= 500:
        phase = 1
    #fell in phase 2, wind (500-1000)
    elif score <= 1000:
        phase = 2
    #reached phase 3, growing force (1000+)
    else:
        phase = 3
    #phase the episode ended in
    return phase

def load_test(methods, test_files=("test_results.csv",)):
    #episodes from evaluate.py (one row per episode) renamed to the names used here
    frames = []
    for test_file in test_files:
        frames.append(pd.read_csv(test_file))
    episodes = pd.concat(frames, ignore_index=True).rename(columns={"net": "run", "len": "length", "ep": "rep", "steps": "score"})
    #a net tested twice keeps its last run
    episodes = episodes.drop_duplicates(["run", "length", "rep"], keep="last")
    method_of, seed_of = {}, {}
    for run in episodes["run"].unique():
        method_of[run], seed_of[run] = split_run(run)
    episodes["method"] = episodes["run"].map(method_of)
    episodes["seed"] = episodes["run"].map(seed_of)
    episodes = episodes[episodes["method"].isin(methods)].copy()
    if episodes.empty:
        raise ValueError(f"No test results for {methods} in {test_files}!")
    episodes["phase"] = episodes["score"].apply(phase_of)
    #one row per test episode
    return episodes

def seed_scores(episodes):
    #per run the mean over lens of the mean over reps, same as the test script total / 30
    per_length = episodes.groupby(["method", "seed", "length"])["score"].mean()
    scores = per_length.groupby(["method", "seed"]).mean().rename("score").reset_index()
    #one row per method and seed
    return scores

def summary(episodes):
    #mean +- std over seeds and how often each phase was the one the pole fell in
    scores = seed_scores(episodes).groupby("method")["score"]
    table = pd.DataFrame({"n_seeds": scores.count(), "mean": scores.mean(), "std": scores.std(), "min": scores.min(), "max": scores.max()})
    phase_share = episodes.groupby("method")["phase"].value_counts(normalize=True).unstack(fill_value=0)
    for phase in (1, 2, 3):
        table[f"fell_phase{phase}"] = phase_share.get(phase, 0.0)
    #one row per method
    return table

def holm(p_values):
    #holm bonferroni: sorted p values times the number of tests left, kept non decreasing
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    max_adjusted = 0.0
    for rank, index in enumerate(order):
        max_adjusted = max(max_adjusted, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, max_adjusted)
    #adjusted p back in the original order
    return adjusted

def compare(scores, pairs):
    #welch t test and mann whitney u on the seed scores of each pair, holm corrected within this family
    rows = []
    for method_a, method_b in pairs:
        scores_a = scores.loc[scores["method"] == method_a, "score"].to_numpy()
        scores_b = scores.loc[scores["method"] == method_b, "score"].to_numpy()
        welch = ttest_ind(scores_a, scores_b, equal_var=False)
        mwu = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        diff = scores_a.mean() - scores_b.mean()
        #effect size, cohens d with the pooled std
        pooled_std = np.sqrt((scores_a.var(ddof=1) + scores_b.var(ddof=1)) / 2)
        rows.append({"a": method_a, "b": method_b, "mean_a": scores_a.mean(), "mean_b": scores_b.mean(), "diff": diff,
                     "cohens_d": diff / pooled_std, "welch_t": welch.statistic, "welch_p": welch.pvalue, "mwu_u": mwu.statistic, "mwu_p": mwu.pvalue})
    table = pd.DataFrame(rows)
    table["welch_p_holm"] = holm(table["welch_p"].to_numpy())
    table["mwu_p_holm"] = holm(table["mwu_p"].to_numpy())
    #one row per pair
    return table

def run_cost(run_dir):
    #env steps from the cfg, val steps and wall clock from the last val row
    cfg = json.loads((run_dir / "cfg.json").read_text())
    val = pd.read_csv(run_dir / "val.csv")
    cost = {"steps": cfg["steps"], "val_steps": int(val["val_steps"].iloc[-1]), "wall_min": val["time"].iloc[-1] / 60}
    #cfg is returned too so the caller can find the specialists
    return cost, cfg

def training_cost(methods, runs_dir="runs"):
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        #unfinished run
        if not (run_dir / "final.pth").exists():
            continue
        method, seed = split_run(run_dir.name)
        #not one of the methods asked for
        if method not in methods:
            continue
        cost, cfg = run_cost(run_dir)
        row = {"method": method, "seed": seed, "train_steps": cost["steps"], "spec_steps": 0, "val_steps": cost["val_steps"], "wall_min": cost["wall_min"]}
        #kickstart students also pay for the specialists of their seed
        if method.startswith("kickstart"):
            for band in BANDS:
                spec_cost, _ = run_cost(Path(runs_dir) / f"spec_{band}{cfg['ks_spec_tag']}_s{seed}")
                row["spec_steps"] += spec_cost["steps"]
                row["val_steps"] += spec_cost["val_steps"]
                row["wall_min"] += spec_cost["wall_min"]
        row["total_steps"] = row["train_steps"] + row["spec_steps"]
        rows.append(row)
    table = pd.DataFrame(rows).drop(columns="seed").groupby("method").agg(["mean", "std"])
    #mean and std over seeds per method
    return table

def best_val_runs(methods, runs_dir="runs"):
    #per method the seed whose saved ckpt had the best own val score, the test is never looked at
    best_runs = {}
    for run_dir in sorted(Path(runs_dir).iterdir()):
        #unfinished run
        if not (run_dir / "final.pth").exists():
            continue
        method, seed = split_run(run_dir.name)
        #not one of the methods asked for
        if method not in methods:
            continue
        val = pd.read_csv(run_dir / "val.csv")
        #best val round of this run, the same one the trainer saved
        best_val = -1
        best_step = 0
        for step, score in zip(val["step"], val["score"]):
            #new best val round
            if score > best_val:
                best_val = score
                best_step = int(step)
        #first run of this method or better than its best seed so far
        if method not in best_runs or best_val > best_runs[method]["best_val"]:
            best_runs[method] = {"seed": seed, "run": run_dir.name, "best_val": best_val, "best_step": best_step}
    picks = pd.DataFrame.from_dict(best_runs, orient="index").sort_index()
    picks.index.name = "method"
    #one row per method
    return picks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, help="run names without _s<seed>")
    parser.add_argument("--baseline", default="uniform", help="every method is tested against this one")
    parser.add_argument("--strategies", nargs="*", default=[], help="these are also tested pairwise")
    parser.add_argument("--controls", nargs="*", default=[], help="strategy:control pairs, eg evolution:evolution_frozen")
    parser.add_argument("--root", default=".", help="dir with runs/ and weights/")
    parser.add_argument("--test", nargs="+", default=["test_results.csv"], help="csvs from evaluate.py")
    parser.add_argument("--out", default="results")
    parser.add_argument("--copy-final", action="store_true", help="copy the best val net per method to weights/final/<method>.pth")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    try:
        episodes = load_test(args.methods, args.test)
        scores = seed_scores(episodes)
        tables = {"summary": summary(episodes), "cost": training_cost(args.methods, f"{args.root}/runs"), "final_picks": best_val_runs(args.methods, f"{args.root}/runs")}
        #separate families so the holm correction isnt spread over every possible pair
        vs_baseline = []
        for method in args.methods:
            #baseline isnt tested against itself
            if method != args.baseline:
                vs_baseline.append((method, args.baseline))
        #every method against the baseline
        if vs_baseline:
            tables["tests_vs_baseline"] = compare(scores, vs_baseline)
        #strategies against each other
        if len(args.strategies) > 1:
            tables["tests_strategies"] = compare(scores, list(combinations(args.strategies, 2)))
        #each strategy against its own control, 3rd family
        if args.controls:
            tables["tests_controls"] = compare(scores, [tuple(pair.split(":")) for pair in args.controls])
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv")
            print(f"\n== {name}\n{table.round(4)}")
        #submitted nets go to weights/final
        if args.copy_final:
            final_dir = Path(args.root) / "weights" / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            for method, pick in tables["final_picks"].iterrows():
                shutil.copy(Path(args.root) / "weights" / f"{pick['run']}.pth", final_dir / f"{method}.pth")
            print(f"\ncopied {len(tables['final_picks'])} nets to {final_dir}")
    except FileNotFoundError as missing_file:
        print(f"Missing file: {missing_file}")
    except ValueError as bad_results:
        print(bad_results)
