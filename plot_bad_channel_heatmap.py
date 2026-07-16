"""Diagnostic heatmap of bad-electrode flags: channel x run, per subject.

This is the companion the full-trace figure cannot be: because that figure robust-z-scores
each channel, amplitude-based faults (dead / hyper-noisy channels) are invisible on it. Here
every (channel, run) cell is coloured by the worst fault detect_bad_channels found, so those
faults are explicit. One panel per subject; run columns within each panel.

    python plot_bad_channel_heatmap.py --dataset faces_basic
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import detect_bad_channels as dbc

# fault severity code per cell (0 clean .. 5 clip), colour + label
CODES = ["clean", "iso", "noisy", "spiky", "dead", "clip"]
COLORS = ["#e9edf2", "#c7d6e5", "#f2c14e", "#e07b39", "#8e6bbf", "#c0392b"]


def _code(tags):
    t = tags
    if "CLIP" in t or "FLAT" in t: return 5
    if "DEAD" in t: return 4
    if "SPIKY" in t: return 3
    if "NOISY" in t: return 2
    if "noisy" in t or "spiky" in t or "iso" in t: return 1
    return 0


def main(dataset):
    df, meta = dbc.detect_all(dataset)
    subs = dbc.DATASETS[dataset]
    fig, axes = plt.subplots(1, len(subs), figsize=(1.7 * len(subs), 12),
                             gridspec_kw=dict(wspace=0.6))
    cmap = ListedColormap(COLORS); norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    for ax, s in zip(np.atleast_1d(axes), subs):
        sd = df[df.subject == s]
        runs = sorted(sd.run.unique()); nch = int(sd.ch.max()) + 1
        grid = np.zeros((nch, len(runs)))
        for _, r in sd.iterrows():
            grid[int(r.ch), runs.index(r.run)] = _code(r.tags)
        ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        title = s
        if meta[s]["drift"]["global_drift"]:
            title += "\n(drift)"
        if meta[s]["segments"]:
            title += "\n(dropout)"
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(runs))); ax.set_xticklabels([f"r{r}" for r in runs], fontsize=7)
        ax.set_yticks(range(0, nch, 5)); ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel("channel", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, CODES, loc="upper center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{dataset}: bad-electrode flags (channel x run)  -- amplitude faults that the "
                 f"z-scored trace figure hides", y=1.05, fontsize=12)
    out = Path(__file__).resolve().parent / f"{dataset}_bad_channel_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="faces_basic", choices=list(dbc.DATASETS))
    main(ap.parse_args().dataset)
