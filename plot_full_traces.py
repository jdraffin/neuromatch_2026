"""Full-run traces for every electrode of every subject, in one tall PNG.

Supersedes scripts/plot_faces_basic_full_traces.py. The old figure robust-z-scored each
channel and hard-clipped it, which made it unusable for spotting amplitude faults: a dead
channel was amplified until it looked normal, a hyper-noisy channel was squashed. This
version keeps the per-channel normalisation (so shape is readable) but makes the figure
*valid for bad-electrode identification*:

  * each channel label carries its NATIVE robust SD, so dead/quiet (tiny SD) and hyper
    (huge SD) channels are identifiable despite the normalisation;
  * channels flagged by detect_bad_channels are drawn in a fault colour and their fault is
    printed next to the label, per run;
  * run boundaries are drawn as vertical lines and whole-montage dropout windows are shaded,
    so per-run faults are visible.

    python plot_full_traces.py --dataset faces_basic
"""
from pathlib import Path
import argparse

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detect_bad_channels as dbc

SPACING = 5.0
SUBJ_GAP = 4 * SPACING
TARGET_PTS = 40000
FAULT_COLOR = {"CLIP": "#c0392b", "FLAT": "#c0392b", "DEAD": "#8e6bbf",
               "SPIKY": "#e07b39", "NOISY": "#d4a017"}


def _fault(tags):
    for key in ("CLIP", "FLAT", "DEAD", "SPIKY", "NOISY"):
        if key in tags:
            return key
    return None


def main(dataset):
    root = dbc.DATA_ROOT / dataset / "data"
    subs = dbc.DATASETS[dataset]
    files = [root / s / f"{s}_faceshouses.mat" for s in subs]

    n_chan_total = 0
    for f in files:
        n_chan_total += sio.loadmat(f, variable_names=["data"])["data"].shape[1]

    width_in, height_in = 54.0, max(20.0, 0.34 * n_chan_total + 1.5 * len(subs))
    print(f"{len(subs)} subjects, {n_chan_total} channels -> {width_in}x{height_in:.0f} in")
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    y, xmax = 0.0, 0.0
    yt_pos, yt_lab, yt_col = [], [], []
    for si, (s, f) in enumerate(zip(subs, files)):
        m = sio.loadmat(f)
        d = m["data"].astype(np.float64); sr = int(m["srate"][0, 0]); n = d.shape[0]
        dd = dbc._notch(d, sr)
        rows, segments, drift = dbc.detect_subject(s, dataset)
        tag_by = {(int(r["run"]), int(r["ch"])): r["tags"] for r in rows}
        runs = dbc.run_bounds(m["stim"])

        stride = max(1, n // TARGET_PTS)
        t = np.arange(n) / sr
        mu = np.median(dd, axis=0)
        sd = np.median(np.abs(dd - mu), axis=0) * 1.4826
        sd[sd == 0] = 1.0
        z = np.clip((dd - mu) / sd, -SPACING * 0.55, SPACING * 0.55)
        xmax = max(xmax, t[-1])
        block_start = y
        col = base_colors[si % 20]

        for ch in range(d.shape[1]):
            # draw run-by-run so a channel bad in only some runs is coloured only there
            ch_fault = None
            for ri, (a, b) in enumerate(runs, start=1):
                sl = slice(a, b, stride)
                fault = _fault(tag_by.get((ri, ch), ""))
                c = FAULT_COLOR[fault] if fault else col
                lw = 0.5 if fault else 0.3
                ax.plot(t[sl], z[a:b:stride, ch] + y, lw=lw, color=c, rasterized=True)
                ch_fault = ch_fault or fault
            lab = f"{s}#{ch} sd={sd[ch]:.0f}" + (f" {ch_fault}" if ch_fault else "")
            yt_pos.append(y); yt_lab.append(lab)
            yt_col.append(FAULT_COLOR[ch_fault] if ch_fault else "black")
            y += SPACING

        for a, b in runs[1:]:                              # run boundaries
            ax.plot([a / sr, a / sr], [block_start - 2, y - SPACING + 2], color="k", lw=0.4, alpha=0.3)
        for seg in segments:                               # dropout windows
            ax.axvspan(seg["t0_s"], seg["t1_s"], ymin=0, ymax=1, color="red", alpha=0.06)
        note = f"{s}\n{d.shape[1]} ch\n{n/sr:.0f}s"
        if drift["global_drift"]:
            note += "\nDRIFT"
        ax.axhline(y + SUBJ_GAP * 0.5, color="k", lw=0.8, alpha=0.4)
        ax.text(-0.010 * t[-1], (block_start + y - SPACING) / 2, note,
                ha="right", va="center", fontsize=14, fontweight="bold", color=col)
        y += SUBJ_GAP

    ax.set_yticks(yt_pos)
    ax.set_yticklabels(yt_lab, fontsize=6)
    for lab, c in zip(ax.get_yticklabels(), yt_col):
        lab.set_color(c)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.set_xlabel("time (s)", fontsize=22); ax.tick_params(axis="x", labelsize=16)
    ax.set_ylim(-SUBJ_GAP, y); ax.set_xlim(0, xmax); ax.invert_yaxis()
    ax.set_title(f"{dataset} full-run traces (60/120/180/240 Hz notch, robust z-scored, offset). "
                 f"Fault-coloured by detect_bad_channels; label shows native SD. "
                 f"purple=dead orange=spiky yellow=noisy red=clip; shaded=dropout", fontsize=20)
    ax.margins(x=0); plt.tight_layout()
    out = Path(__file__).resolve().parent / f"{dataset}_all_subjects_full_traces.png"
    fig.savefig(out, dpi=96)
    print("saved", out, f"{Path(out).stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="faces_basic", choices=list(dbc.DATASETS))
    main(ap.parse_args().dataset)
