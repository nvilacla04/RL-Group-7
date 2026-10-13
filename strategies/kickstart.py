import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.config import Cfg
from test_script import QNetwork

#specialist bands from short to long poles, the edges are in ks_edges
BANDS = ("short", "medium", "long")

#strategy 2: kickstarting from specialists (schmitt et al 2018)
#3 dqns are trained on short, medium and long poles and frozen, then the student (normal dqn on all lens)
#trains on loss = l_td + lam_t * l_ks, where l_ks pulls it towards the specialist of each transitions band
#lam_t goes to 0 so the student ends as a pure dqn
#plugs into the core without changing it, the kickstart class below is the aux loss agent.update calls every update
@dataclass
class KSCfg(Cfg):
    ks_edges: tuple = (0.4, 0.87, 1.33, 1.8)
    #in the pilot all specialists passed 0.9 val by 130k steps
    ks_spec_steps: int = 150_000
    #false = student gets the same steps as the baselines and specialists come on top
    #true = specialist steps come out of the student budget so the total is steps
    ks_count_spec: bool = False
    #suffix of the specialist runs the student loads
    ks_spec_tag: str = ""
    #kl on q softmaxes, mse only for comparison
    ks_loss: str = "kl"
    #temp from rusu et al 2016
    ks_tau: float = 0.01
    #pilot with lam0 = 0 gave l_td / l_ks of about 0.02 to 0.04 early on so lam0 * l_ks is about l_td
    ks_lam0: float = 0.03
    #frac of student steps where lam hits 0
    ks_decay: float = 0.5
    #width of the blend between neighbour teachers around each edge, 0 = hard band lookup
    ks_blend: float = 0.0
    #env steps per row in ks.csv
    ks_log_every: int = 1000

def band_bounds(cfg, band):
    #lo and hi len of one band
    band_index = BANDS.index(band)
    bounds = (cfg.ks_edges[band_index], cfg.ks_edges[band_index + 1])
    #lo and hi returned as a pair
    return bounds

def band_val_lens(cfg, band):
    #own val lens inside the band so each specialist is picked on its own lens
    lo, hi = band_bounds(cfg, band)
    lens_in_band = []
    for length in cfg.val_lens:
        #len is in the band
        if lo <= length <= hi:
            lens_in_band.append(length)
    #tuple like the val_lens in the cfg
    return tuple(lens_in_band)

def load_teacher(path):
    #frozen copy of a specialist, eval mode and no grads
    teacher = QNetwork(4, 2)
    teacher.load_state_dict(torch.load(path, weights_only=True))
    teacher.eval().requires_grad_(False)
    #ready to use as a teacher
    return teacher

class Kickstart():

    def __init__(self, cfg, teacher_paths, log_path=None):
        #settings that would break the band lookup or the logging are stopped here
        if len(cfg.ks_edges) - 1 != len(BANDS) or len(teacher_paths) != len(BANDS):
            raise ValueError("Need exactly one teacher and one band per entry in BANDS!")
        if cfg.ks_loss not in ("kl", "mse"):
            raise ValueError(f"Unknown ks_loss {cfg.ks_loss}, use kl or mse!")
        if cfg.ks_blend >= min(np.diff(cfg.ks_edges)):
            raise ValueError("ks_blend has to be smaller than the narrowest band!")
        if cfg.ks_log_every % cfg.train_freq != 0:
            raise ValueError("ks_log_every has to be a multiple of train_freq!")
        self.cfg = cfg
        self.teachers = []
        for path in teacher_paths:
            self.teachers.append(load_teacher(path))
        #inner edges split the len range into the bands
        self.inner_edges = torch.tensor(cfg.ks_edges[1:-1], dtype=torch.float32)
        self.log_path = log_path
        #old log from a crashed run is removed so rows dont get doubled
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).unlink(missing_ok=True)
        self.reset_window()

    def lam_at(self, step):
        #linear decay from lam0 to 0 at ks_decay of the run, pure dqn after that
        decay_steps = self.cfg.ks_decay * self.cfg.steps
        lam = self.cfg.ks_lam0 * max(0.0, 1 - step / decay_steps)
        #weight of l_ks at this step
        return lam

    def band_weights(self, lengths):
        #weight of each teacher per transition
        if self.cfg.ks_blend > 0:
            #near an edge blend linearly between the 2 neighbours, frac above goes 0 to 1 across the blend width
            frac_above = ((lengths.unsqueeze(1) - self.inner_edges) / self.cfg.ks_blend + 0.5).clamp(0, 1)
        else:
            #hard lookup, frac above is 0 or 1
            frac_above = (lengths.unsqueeze(1) >= self.inner_edges).float()
        #teacher i gets frac above edge i-1 minus frac above edge i, so one hot without blending
        ones = torch.ones(len(lengths), 1)
        zeros = torch.zeros(len(lengths), 1)
        weights = torch.cat([ones, frac_above], 1) - torch.cat([frac_above, zeros], 1)
        #one row per transition, one column per teacher
        return weights

    @torch.no_grad()
    def teacher_q(self, states, lengths):
        #q of every teacher stacked to (batch, teacher, action)
        all_teacher_q = []
        for teacher in self.teachers:
            all_teacher_q.append(teacher(states))
        all_teacher_q = torch.stack(all_teacher_q, 1)
        #mixed with the band weights so each transition gets its own teacher
        mixed_q = (self.band_weights(lengths).unsqueeze(2) * all_teacher_q).sum(1)
        #teacher q per transition, no grads since teachers are frozen
        return mixed_q

    def ks_loss(self, student_q, teacher_q):
        #kl, the default
        if self.cfg.ks_loss == "kl":
            #kl(teacher || student) on softmax(q / tau), rusu et al only put tau on the teacher
            #but the student also needs real q values for the td loss so it gets the same tau
            tau = self.cfg.ks_tau
            loss = F.kl_div(F.log_softmax(student_q / tau, 1), F.log_softmax(teacher_q / tau, 1), log_target=True, reduction="batchmean")
        else:
            #mse: plain regression on the q values
            loss = F.mse_loss(student_q, teacher_q)
        #l_ks before the lam weight
        return loss

    def __call__(self, batch, student_q, step):
        #called by agent.update with the batch, the student q values and the env step
        teacher_q = self.teacher_q(batch["s"], batch["l"])
        ks_loss = self.ks_loss(student_q, teacher_q)
        lam = self.lam_at(step)
        self.track(batch["l"], student_q.detach(), teacher_q, ks_loss.item(), lam)
        #window is full so its written to ks.csv
        if step % self.cfg.ks_log_every == 0:
            self.flush(step)
        weighted_loss = lam * ks_loss
        #gets added to the td loss in agent.update
        return weighted_loss

    def reset_window(self):
        #sums over the current log window
        self.window = {"n_upd": 0, "ks": 0.0, "lam": 0.0, "lam_ks": 0.0}
        self.agree_sum = np.zeros(len(BANDS))
        self.band_count = np.zeros(len(BANDS))

    def track(self, lengths, student_q, teacher_q, ks_loss, lam):
        self.window["n_upd"] += 1
        self.window["ks"] += ks_loss
        self.window["lam"] += lam
        self.window["lam_ks"] += lam * ks_loss
        #student teacher greedy action agreement per band, hard band even when blending
        band = (lengths.unsqueeze(1) >= self.inner_edges).sum(1).numpy()
        same_action = (student_q.argmax(1) == teacher_q.argmax(1)).float().numpy()
        self.agree_sum += np.bincount(band, weights=same_action, minlength=len(BANDS))
        self.band_count += np.bincount(band, minlength=len(BANDS))

    def flush(self, step):
        #one row of window means, n_upd is kept so windows can be merged into val rounds later
        n_upd = self.window["n_upd"]
        row = {"step": step, "n_upd": n_upd}
        for key in ("ks", "lam", "lam_ks"):
            row[key] = self.window[key] / n_upd
        for band_index, band in enumerate(BANDS):
            #band had transitions in this window
            if self.band_count[band_index] > 0:
                row[f"agree_{band}"] = self.agree_sum[band_index] / self.band_count[band_index]
            #no transitions from this band in the window
            else:
                row[f"agree_{band}"] = float("nan")
        #only written when a log path was given
        if self.log_path is not None:
            write_header = not Path(self.log_path).exists()
            with open(self.log_path, "a", newline="") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                #header only on the first row
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        self.reset_window()
