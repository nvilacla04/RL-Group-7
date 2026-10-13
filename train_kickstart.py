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
    #eg spec_short_s0
    return name

def student_name(seed, tag=""):
    name = f"kickstart{tag}_s{seed}"
    #eg kickstart_s0
    return name

def run_specialist(band, seed, overrides, tag, root):
    #uniform baseline on one band only, ckpt picked on the val lens inside that band
    cfg = KSCfg(seed=seed).set(overrides)
    lo, hi = band_bounds(cfg, band)
    spec_cfg = replace(cfg, name=spec_name(band, seed, tag), steps=cfg.ks_spec_steps, lo=lo, hi=hi, val_lens=band_val_lens(cfg, band))
    result = train(spec_cfg, Uniform(lo, hi), out=f"{root}/runs", wdir=f"{root}/weights")
    #summary dict from the trainer
    return result

def run_student(seed, overrides, tag, root):
    cfg = KSCfg(name=student_name(seed, tag), seed=seed).set(overrides)
    #specialist steps come out of the budget so the total matches the baselines
    if cfg.ks_count_spec:
        cfg = replace(cfg, steps=cfg.steps - len(BANDS) * cfg.ks_spec_steps)
    teacher_paths = []
    for band in BANDS:
        teacher_path = Path(root) / "weights" / f"{spec_name(band, seed, cfg.ks_spec_tag)}.pth"
        #student cant start without all 3 teachers
        if not teacher_path.exists():
            raise FileNotFoundError(f"Specialist {teacher_path} is missing, run --stage spec first!")
        teacher_paths.append(teacher_path)
    kickstart = Kickstart(cfg, teacher_paths, Path(root) / "runs" / cfg.name / "ks.csv")
    #same trainer as the baselines, the kickstart loss goes in as aux
    result = train(cfg, Uniform(cfg.lo, cfg.hi), aux=kickstart, out=f"{root}/runs", wdir=f"{root}/weights")
    #summary dict from the trainer
    return result

def job_name(job):
    stage, band, seed, overrides, tag, root = job
    #specialist job
    if stage == "spec":
        name = spec_name(band, seed, tag)
    #student job
    else:
        name = student_name(seed, tag)
    #run name that the job will write to
    return name

def run(job):
    #job = (stage, band, seed, cfg overrides, tag, root dir)
    stage, band, seed, overrides, tag, root = job
    torch.set_num_threads(1)
    #specialist job
    if stage == "spec":
        result = run_specialist(band, seed, overrides, tag, root)
    #student job
    else:
        result = run_student(seed, overrides, tag, root)
    #summary dict from the trainer
    return result

def launch(jobs, procs=1):
    #runs that already finished are skipped so it can just be rerun
    todo = []
    for job in jobs:
        #no final net yet so it still has to run
        if not (Path(job[5]) / "runs" / job_name(job) / "final.pth").exists():
            todo.append(job)
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run")
    #more than one job so they run in parallel
    if procs > 1 and len(todo) > 1:
        with Pool(min(procs, len(todo))) as pool:
            for result in pool.imap_unordered(run, todo):
                print(result, flush=True)
    #one at a time
    else:
        for job in todo:
            print(run(job), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("spec", "student", "all"), required=True)
    parser.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--set", nargs="*", default=[], help="cfg overrides, eg ks_lam0=0.01 ks_spec_steps=60000")
    parser.add_argument("--tag", default="", help="suffix for run names")
    parser.add_argument("--root", default=".", help="dir for runs/ and weights/")
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()
    try:
        #students need the specialists, so with all the spec stage finishes first
        if args.stage in ("spec", "all"):
            spec_jobs = []
            for seed in args.seeds:
                for band in args.bands:
                    spec_jobs.append(("spec", band, seed, args.set, args.tag, args.root))
            launch(spec_jobs, args.procs)
        #student stage
        if args.stage in ("student", "all"):
            student_overrides = list(args.set)
            #with all the students load the specialists that were just trained under the same tag
            if args.stage == "all":
                for override in args.set:
                    #spec tag was set by hand so its kept
                    if override.startswith("ks_spec_tag="):
                        break
                else:
                    #no spec tag set so the specialists from this call are used
                    student_overrides.append(f"ks_spec_tag={args.tag}")
            student_jobs = []
            for seed in args.seeds:
                student_jobs.append(("student", None, seed, student_overrides, args.tag, args.root))
            launch(student_jobs, args.procs)
    except FileNotFoundError as missing_specialist:
        print(missing_specialist)
    except AttributeError as unknown_setting:
        print(f"Unknown cfg setting: {unknown_setting}")
    except ValueError as bad_setting:
        print(bad_setting)
