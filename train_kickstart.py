import argparse
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import torch

from core.samplers import Uniform
from core.trainer import train
from strategies.kickstart import BANDS, KSCfg, Kickstart, band_bounds, band_val_lens


#launcher for strategy 2, same idea as train.py but in 2 stages
#spec: 3 specialists per seed, student: kickstarted dqn that loads the specialists of its own seed
#eg python train_kickstart.py --stage all --seeds 0 1 2 3 4 --procs 6
def spec_name(band, seed, tag=""):
    name = f"spec_{band}{tag}_s{seed}"
    return name


def student_name(seed, tag=""):
    name = f"kickstart{tag}_s{seed}"
    return name


def run_specialist(band, seed, overrides, tag, root):
    #uniform baseline on one band only, ckpt picked on the val lens inside that band
    cfg = KSCfg(seed=seed).set(overrides)
    lo, hi = band_bounds(cfg, band)
    spec_cfg = replace(cfg, name=spec_name(band, seed, tag), steps=cfg.ks_spec_steps, lo=lo, hi=hi, val_lens=band_val_lens(cfg, band))
    result = train(spec_cfg, Uniform(lo, hi), out=f"{root}/runs", wdir=f"{root}/weights")
    return result


def run_student(seed, overrides, tag, root):
    cfg = KSCfg(name=student_name(seed, tag), seed=seed).set(overrides)
    if cfg.ks_count_spec:
        #specialist steps come out of the budget so the total matches the baselines
        cfg = replace(cfg, steps=cfg.steps - len(BANDS) * cfg.ks_spec_steps)
    teacher_paths = []
    for band in BANDS:
        teacher_path = Path(root) / "weights" / f"{spec_name(band, seed, cfg.ks_spec_tag)}.pth"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Specialist {teacher_path} is missing, run --stage spec first!")
        teacher_paths.append(teacher_path)
    kickstart = Kickstart(cfg, teacher_paths, Path(root) / "runs" / cfg.name / "ks.csv")
    result = train(cfg, Uniform(cfg.lo, cfg.hi), aux=kickstart, out=f"{root}/runs", wdir=f"{root}/weights")
    return result


def job_name(job):
    stage, band, seed, overrides, tag, root = job
    if stage == "spec":
        name = spec_name(band, seed, tag)
    else:
        name = student_name(seed, tag)
    return name


def run(job):
    #job = (stage, band, seed, cfg overrides, tag, root dir)
    stage, band, seed, overrides, tag, root = job
    torch.set_num_threads(1)
    if stage == "spec":
        result = run_specialist(band, seed, overrides, tag, root)
    else:
        result = run_student(seed, overrides, tag, root)
    return result


def launch(jobs, procs=1):
    #skips runs that already finished so it can just be rerun
    todo = []
    for job in jobs:
        if not (Path(job[5]) / "runs" / job_name(job) / "final.pth").exists():
            todo.append(job)
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run")
    if procs > 1 and len(todo) > 1:
        with Pool(min(procs, len(todo))) as pool:
            for result in pool.imap_unordered(run, todo):
                print(result, flush=True)
    else:
        for job in todo:
            print(run(job), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("spec", "student", "all"), required=True)
    p.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--set", nargs="*", default=[], help="cfg overrides, eg ks_lam0=0.01 ks_spec_steps=60000")
    p.add_argument("--tag", default="", help="suffix for run names")
    p.add_argument("--root", default=".", help="dir for runs/ and weights/")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    #students need the specialists, so with all the spec stage finishes first
    if a.stage in ("spec", "all"):
        launch([("spec", band, seed, a.set, a.tag, a.root) for seed in a.seeds for band in a.bands], a.procs)
    if a.stage in ("student", "all"):
        student_overrides = list(a.set)
        #with all the students load the specialists that were just trained under the same tag
        if a.stage == "all" and not any(kv.startswith("ks_spec_tag=") for kv in a.set):
            student_overrides.append(f"ks_spec_tag={a.tag}")
        launch([("student", None, seed, student_overrides, a.tag, a.root) for seed in a.seeds], a.procs)
