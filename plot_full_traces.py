"""Full-run traces for every electrode of every subject, in one tall PNG.

Offset stack, one coloured block per subject, every channel labelled on both sides with its
native robust SD. Above each subject block sits a trial strip: one tick per trial, numbered
run-locally (0-based, restarting each run) to match EXCLUDE_TRIALS in preprocessing_pipeline.py.
Two scalings (push both):
  * z-scored (default): each channel robust-z-scored to fill its lane -> read the PATTERN.
  * raw (--no-zscore): per-subject common gain -> amplitude is honest, so tiny-variance
    channels are visibly small and loud channels fill their lane.

    python plot_full_traces.py --dataset faces_basic --both
"""
from pathlib import Path
import argparse

import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import preprocessing_pipeline as pp    # shading is driven by the pipeline's own exclusion tables

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset"
DATASETS = {
    "faces_basic": ["aa", "ap", "ca", "de", "fp", "ha", "ja", "jm", "jt", "mv", "rn", "rr", "wc", "zt"],
    "faces_noise": ["ap", "ca", "ha", "ja", "mv", "wc", "zt"],
}
NOTCH_FREQS = (60.0, 120.0, 180.0, 240.0)
NOTCH_Q = 30.0

SPACING = 5.0
SUBJ_GAP = 4 * SPACING
TARGET_PTS = 40000
LABELSIZE = 13          # electrode-label font size (left and right)
ROW_INCHES = 0.36       # vertical space per channel; must fit LABELSIZE without collisions
TRIAL_LABEL_EVERY = 5   # tick every trial; print the number every Nth


def _notch(x, sf):
    for f0 in NOTCH_FREQS:
        if f0 < sf / 2:
            b, a = iirnotch(f0, NOTCH_Q, sf)
            x = filtfilt(b, a, x, axis=0)
    return x


def _run_bounds(stim, min_gap_samples=2000):
    """Sample [start, end) of each run, from the stim==0 gaps; drop the all-zero tail."""
    stim = np.asarray(stim).ravel().astype(int)
    chg = np.where(np.diff(stim) != 0)[0] + 1
    starts = np.concatenate([[0], chg])
    ends = np.concatenate([chg, [len(stim)]])
    gaps = [(s, e) for s, e in zip(starts, ends)
            if stim[s] == 0 and (e - s) >= min_gap_samples and 1000 < s < len(stim) - 1000]
    bnds = [0] + [(s + e) // 2 for s, e in gaps] + [len(stim)]
    segs = [(bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)]
    return [(a, b) for a, b in segs if ((stim[a:b] >= 1) & (stim[a:b] <= 100)).any()]


def _trial_onsets(stim):
    """Sample index of every faceshouses stimulus onset (stim goes to a 1..100 code)."""
    v = np.asarray(stim).ravel().astype(int)
    is_stim = (v >= 1) & (v <= 100)
    return np.where(is_stim[1:] & (v[1:] != v[:-1]))[0] + 1


def _draw_trial_strip(ax, stim, sr, runs, block_start, col):
    """Trial-number strip above a subject block: a tick per trial, numbered run-locally.

    Numbers restart at 0 each run and are 0-based, matching EXCLUDE_TRIALS in
    preprocessing_pipeline.py ({subject: {run: trials}}), so a number read off this figure can
    be pasted straight in. Deliberately no face/house marking - stay blind to condition when
    judging which trials to exclude.
    """
    onsets = _trial_onsets(stim)
    y_tick_top, y_tick_bot = block_start - 4.0, block_start - 2.2
    for ri, (a, b) in enumerate(runs):
        in_run = onsets[(onsets >= a) & (onsets < b)]
        if not len(in_run):
            continue
        ts = in_run / sr
        ax.vlines(ts, y_tick_top, y_tick_bot, color=col, lw=0.4, alpha=0.75)
        for k, t0 in enumerate(ts):
            if k % TRIAL_LABEL_EVERY == 0:
                ax.text(t0, y_tick_top - 0.4, str(k), ha="center", va="bottom",
                        fontsize=7, color=col, rotation=90)
        ax.text((ts[0] + ts[-1]) / 2, block_start - 6.8, f"run {ri}  ({len(ts)} trials)",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=col)


def _trial_grid(stim, sr, runs):
    """Per-trial (run, run-local index, onset time) across a subject, plus run lengths."""
    onsets = _trial_onsets(stim)
    r, k, t = [], [], []
    for ri, (a, b) in enumerate(runs):
        o = onsets[(onsets >= a) & (onsets < b)] / sr
        r.append(np.full(len(o), ri)); k.append(np.arange(len(o))); t.append(o)
    r, k, t = np.concatenate(r), np.concatenate(k), np.concatenate(t)
    run_last = {ri: int(k[r == ri].max()) for ri in np.unique(r)}

    return r, k, t, run_last


def _excl_spans(where, runs, sr, onset_of, dur):
    """(t0, t1) time spans for an exclusion spec ("all" | [runs] | {run: trials}).

    "start"/"end" in a range reach the run's own boundary rather than its first/last trial, so
    the pre-first-trial and post-last-trial intervals get shaded too - which is the whole point
    of writing them instead of a bare 0 / last index.
    """
    if isinstance(where, str):                             # "all" -> every run, end to end
        return [(runs[0][0] / sr, runs[-1][1] / sr)]
    items = where.items() if isinstance(where, dict) else [(r, "all") for r in where]
    spans = []
    for run, trials in items:
        if not (0 <= run < len(runs)):
            continue
        a, b = runs[run][0] / sr, runs[run][1] / sr
        if isinstance(trials, str):                        # whole run
            spans.append((a, b))
            continue
        for item in trials:
            if isinstance(item, (tuple, list)):
                s, e = item
                t0 = a if s == "start" else onset_of[(run, int(s))]
                t1 = b if e == "end" else onset_of[(run, int(e))] + dur
            else:
                t0 = onset_of[(run, int(item))]
                t1 = t0 + dur
            spans.append((t0, t1))

    return spans


def _shade_exclusions(ax, subj, stim, sr, runs, block_start, nch):
    """Faint red over everything the pipeline excludes for this subject.

    Read straight from preprocessing_pipeline's tables, so the figure always shows what the
    pipeline actually does. Electrode-level exclusions shade only that channel's lane; whole-
    trial and whole-run exclusions shade the full block height (every electrode loses them).
    """
    r, k, t, _ = _trial_grid(stim, sr, runs)
    dur = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    onset_of = {(int(ri), int(ki)): float(ti) for ri, ki, ti in zip(r, k, t)}
    lane = dict(color="red", lw=0, zorder=0)

    for elec, where in pp.EXCLUDE_CHANNELS.get(subj, {}).items():
        if not (0 <= elec < nch):
            continue
        y = block_start + elec * SPACING - SPACING / 2
        for t0, t1 in _excl_spans(where, runs, sr, onset_of, dur):
            ax.add_patch(Rectangle((t0, y), t1 - t0, SPACING, alpha=0.20, **lane))

    whole = dict(pp.EXCLUDE_TRIALS.get(subj, {}))          # whole-trial and whole-run drops
    whole.update({run: "all" for run in pp.EXCLUDE_RUNS.get(subj, [])})
    y0, h = block_start - SPACING / 2, (nch - 1) * SPACING + SPACING
    for t0, t1 in _excl_spans(whole, runs, sr, onset_of, dur):
        ax.add_patch(Rectangle((t0, y0), t1 - t0, h, alpha=0.10, **lane))


def main(dataset, zscore=True):
    root = DATA_ROOT / dataset / "data"
    subs = DATASETS[dataset]
    files = [root / s / f"{s}_faceshouses.mat" for s in subs]

    n_chan_total = 0
    for f in files:
        n_chan_total += sio.loadmat(f, variable_names=["data"])["data"].shape[1]

    width_in, height_in = 54.0, max(20.0, ROW_INCHES * n_chan_total + 1.5 * len(subs))
    print(f"{len(subs)} subjects, {n_chan_total} channels -> {width_in}x{height_in:.0f} in")
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    y, xmax = 0.0, 0.0
    yt_pos, yt_lab = [], []
    for si, (s, f) in enumerate(zip(subs, files)):
        m = sio.loadmat(f)
        d = m["data"].astype(np.float64); sr = int(m["srate"][0, 0]); n = d.shape[0]
        dd = _notch(d, sr)
        runs = _run_bounds(m["stim"])

        stride = max(1, n // TARGET_PTS)
        t = np.arange(n) / sr
        mu = np.median(dd, axis=0)
        sd = np.median(np.abs(dd - mu), axis=0) * 1.4826   # per-channel robust SD (native units)
        sd_safe = np.where(sd == 0, 1.0, sd)
        if zscore:
            # per-CHANNEL gain: every channel fills its lane -> pattern readable, amplitude hidden
            z = np.clip((dd - mu) / sd_safe, -SPACING * 0.55, SPACING * 0.55)
        else:
            # per-SUBJECT common gain: one gain for all channels so amplitude is honest.
            # Scale to the 90th-percentile SD so loud channels fill their lane without
            # overrunning neighbours, and a tiny-variance channel is visibly small.
            gain = SPACING * 0.55 / np.percentile(sd_safe, 90)
            z = np.clip((dd - mu) * gain, -SPACING * 0.62, SPACING * 0.62)
        xmax = max(xmax, t[-1])
        block_start = y
        col = base_colors[si % 20]

        for ch in range(d.shape[1]):
            ax.plot(t[::stride], z[::stride, ch] + y, lw=0.3, color=col, rasterized=True)
            yt_pos.append(y); yt_lab.append(f"{s}#{ch} sd={sd[ch]:.0f}")
            y += SPACING

        _draw_trial_strip(ax, m["stim"], sr, runs, block_start, col)
        _shade_exclusions(ax, s, m["stim"], sr, runs, block_start, d.shape[1])
        for a, b in runs[1:]:                              # run boundaries
            ax.plot([a / sr, a / sr], [block_start - 2, y - SPACING + 2], color="k", lw=0.4, alpha=0.3)
        ax.axhline(y + SUBJ_GAP * 0.5, color="k", lw=0.8, alpha=0.4)
        ax.text(-0.010 * t[-1], (block_start + y - SPACING) / 2,
                f"{s}\n{d.shape[1]} ch\n{n/sr:.0f}s",
                ha="right", va="center", fontsize=14, fontweight="bold", color=col)
        y += SUBJ_GAP

    ax.set_ylim(-SUBJ_GAP, y); ax.set_xlim(0, xmax); ax.invert_yaxis()
    # electrode labels on BOTH sides, enlarged
    ax.set_yticks(yt_pos); ax.set_yticklabels(yt_lab, fontsize=LABELSIZE)
    ax.tick_params(axis="y", length=2, pad=1)
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())                            # mirror inverted range
    axr.set_yticks(yt_pos); axr.set_yticklabels(yt_lab, fontsize=LABELSIZE)
    axr.tick_params(axis="y", length=2, pad=1)
    ax.set_xlabel("time (s)", fontsize=22); ax.tick_params(axis="x", labelsize=16)
    scale = "robust z-scored per channel" if zscore else "per-subject common gain"
    ax.set_title(f"{dataset} full-run traces (60/120/180/240 Hz notch, {scale}, offset) - "
                 f"trial ticks above each block, numbered 0-based within each run",
                 fontsize=20)
    ax.margins(x=0); plt.tight_layout()
    suffix = "zscored" if zscore else "raw"
    out = Path(__file__).resolve().parent / f"{dataset}_all_subjects_full_traces_{suffix}.png"
    fig.savefig(out, dpi=96 if zscore else 84)             # raw kept <100 MB (GitHub hard limit)
    print("saved", out, f"{Path(out).stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="faces_basic", choices=list(DATASETS))
    ap.add_argument("--no-zscore", dest="zscore", action="store_false",
                    help="per-subject common gain instead of per-channel z-score")
    ap.add_argument("--both", action="store_true", help="render both zscored and raw")
    args = ap.parse_args()
    if args.both:
        main(args.dataset, zscore=True)
        main(args.dataset, zscore=False)
    else:
        main(args.dataset, zscore=args.zscore)
