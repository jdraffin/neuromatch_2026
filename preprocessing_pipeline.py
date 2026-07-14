"""Shared preprocessing pipeline for the Miller faces/houses ECoG dataset.

One canonical entry point so every team member preprocesses and *validates* the data the
same way. Produces MNE Epochs (with metadata) and the agreed cross-validation splits.

Design decisions (see notes at bottom):
  * Filtering: line-noise NOTCH at 60/120/180/240 Hz, plus an optional band-pass.
    (The raw data already has a gentle 1-pole hardware band-pass 0.15-200 Hz; a sharper
     band-pass is optional and off by default.)
  * VALIDATION SPLIT = leave-one-IMAGE-out (group by stimulus id). The 100 images repeat
    in identical order across all 3 runs, so leave-one-run-out lets the model see every
    test image in training (image-identity leakage) -> it measures memorisation, not
    generalisation. Leave-one-image-out holds out ALL repetitions of held-out images and
    is the honest test of category generalisation. leave-one-run-out is also provided for
    cross-run stability analyses.

Usage
-----
    from preprocessing_pipeline import preprocess, leave_one_image_out, leave_one_run_out
    epochs = preprocess("ca", task="faceshouses")          # mne.Epochs with .metadata
    X = epochs.get_data()                                   # (n_trials, n_ch, n_times)
    y = (epochs.metadata["category"] == "face").astype(int).to_numpy()
    for train_idx, test_idx in leave_one_image_out(epochs): ...

    # CLI: build + save shareable epochs for a subject
    python preprocessing_pipeline.py --subject ca --task faceshouses --save
"""

from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut

mne.set_log_level("ERROR")

# Default location of the dataset (edit for your machine, or pass root=...).
DATA_ROOT = Path(__file__).resolve().parent.parent / "Project" / "dataset" / "faces_noise" / "data"
SUBJECTS = ["ap", "ca", "ha", "ja", "mv", "wc", "zt"]

NOTCH = (60, 120, 180, 240)     # line-noise harmonics
BANDPASS = None                 # e.g. (1.0, 200.0) to add a sharper band-pass; None = off
WIN_MS = (0, 400)               # epoch window relative to stimulus onset


# --------------------------------------------------------------------------- raw + filter
def to_raw(mat_path, notch=NOTCH, bandpass=BANDPASS):
    """`.mat` -> filtered mne.io.RawArray (ECoG channels + a STIM channel), and the mat dict.

    Line-noise notch and optional band-pass are applied to the ECoG channels only.
    """
    m = loadmat(mat_path)
    sf = float(np.ravel(m["srate"])[0])
    data = np.asarray(m["data"], dtype=float).T                 # (channel, time)
    stim = np.ravel(m["stim"]).astype(float)

    n_ch = data.shape[0]
    names = [f"ecog{i}" for i in range(n_ch)] + ["STIM"]
    types = ["ecog"] * n_ch + ["stim"]
    raw = mne.io.RawArray(np.vstack([data, stim[None, :]]), mne.create_info(names, sf, types))

    picks = mne.pick_types(raw.info, ecog=True)
    if notch:
        raw.notch_filter([f for f in notch if f < sf / 2], picks=picks)
    if bandpass:
        raw.filter(bandpass[0], bandpass[1], picks=picks)
    return raw, m


# --------------------------------------------------------------------------- run detection
def _run_index(stim, onsets, min_gap_samples=2000):
    """Assign each stimulus onset to a run, using the ~9 s stim==0 gaps between runs.

    Returns an int run index per onset (0-based). If no gaps are found (single continuous
    recording, e.g. subject `ha`), everything is run 0.
    """
    stim = np.asarray(stim).astype(int)
    chg = np.where(np.diff(stim) != 0)[0] + 1
    starts = np.concatenate([[0], chg])
    ends = np.concatenate([chg, [len(stim)]])
    gap_starts = [s for s, e in zip(starts, ends)
                  if stim[s] == 0 and (e - s) >= min_gap_samples and 1000 < s < len(stim) - 1000]
    run = np.zeros(len(onsets), dtype=int)
    for g in gap_starts:
        run[onsets > g] += 1
    return run


# --------------------------------------------------------------------------- epoching
def preprocess(subject, task="faceshouses", root=DATA_ROOT,
               notch=NOTCH, bandpass=BANDPASS, win_ms=WIN_MS):
    """Load, filter, and epoch one subject/task into mne.Epochs with metadata.

    metadata columns:
      faceshouses: category ('face'/'house'), stim_id (1..100), run (0..2)
      fhnoisy    : category, coherence (0..100 % noise), run (always 0; noise randomised)
    """
    raw, m = to_raw(Path(root) / subject / f"{subject}_{task}.mat", notch, bandpass)
    sf = raw.info["sfreq"]
    stim = np.ravel(m["stim"]).astype(int)
    events = mne.find_events(raw, stim_channel="STIM", consecutive=True)

    if task == "faceshouses":
        events = events[(events[:, 2] >= 1) & (events[:, 2] <= 100)]
        codes = events[:, 2]
        meta = pd.DataFrame({
            "stim_id": codes,
            "category": np.where(codes <= 50, "house", "face"),
            "run": _run_index(stim, events[:, 0]),
        })
    elif task == "fhnoisy":
        events = events[events[:, 2] >= 1]                      # trial-counter onsets
        tr_fh = np.ravel(m["tr_fh"]).astype(int)
        tr_coh = np.ravel(m["tr_coh"]).astype(int)
        n = min(len(events), len(tr_fh))
        events, tr_fh, tr_coh = events[:n], tr_fh[:n], tr_coh[:n]
        meta = pd.DataFrame({
            "stim_id": np.arange(1, n + 1),                     # each noisy trial is unique
            "category": np.where(tr_fh == 1, "house", "face"),
            "coherence": tr_coh,
            "run": 0,
        })
    else:
        raise ValueError(task)

    tmin, tmax = win_ms[0] / 1000.0, win_ms[1] / 1000.0 - 1.0 / sf
    ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None,
                    preload=True, picks="ecog", metadata=meta)
    return ep


# --------------------------------------------------------------------------- splits
def _y(epochs):
    return (epochs.metadata["category"].to_numpy() == "face").astype(int)


def leave_one_image_out(epochs, n_splits=5, seed=0):
    """CANONICAL validation split: group by stimulus id, stratified by category.

    Holds out all repetitions of the held-out images -> tests generalisation to NOVEL
    images. Returns a list of (train_idx, test_idx).
    """
    y = _y(epochs)
    groups = epochs.metadata["stim_id"].to_numpy()
    Xd = np.zeros((len(y), 1))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(Xd, y, groups))


def leave_one_run_out(epochs):
    """Cross-run stability split (group by run). Not a generalisation test (images repeat
    across runs), but useful for stability / novelty analyses. faceshouses only."""
    groups = epochs.metadata["run"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("only one run present (e.g. subject ha) - leave-one-run-out N/A")
    Xd = np.zeros((len(groups), 1))
    return list(LeaveOneGroupOut().split(Xd, groups=groups))


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="ca")
    ap.add_argument("--task", default="faceshouses", choices=["faceshouses", "fhnoisy"])
    ap.add_argument("--bandpass", nargs=2, type=float, default=None, help="e.g. --bandpass 1 200")
    ap.add_argument("--save", action="store_true", help="write <subject>_<task>-epo.fif")
    args = ap.parse_args()

    ep = preprocess(args.subject, args.task, bandpass=tuple(args.bandpass) if args.bandpass else None)
    print(ep)
    print(ep.metadata.head())
    print(f"n_trials={len(ep)}  n_channels={len(ep.ch_names)}  times={ep.times[0]:.3f}..{ep.times[-1]:.3f}s")
    if args.task == "faceshouses":
        splits = leave_one_image_out(ep)
        print(f"leave-one-image-out: {len(splits)} folds; "
              f"fold0 train={len(splits[0][0])} test={len(splits[0][1])} "
              f"(test images all unseen in train)")
        # sanity: no image id shared between train and test of fold 0
        sid = ep.metadata["stim_id"].to_numpy()
        tr, te = splits[0]
        print(f"fold0 image overlap train/test = {len(set(sid[tr]) & set(sid[te]))} (should be 0)")
    if args.save:
        out = Path(__file__).resolve().parent / f"{args.subject}_{args.task}-epo.fif"
        ep.save(out, overwrite=True)
        print(f"saved {out}")

# -----------------------------------------------------------------------------------------
# Notes for the team
# -----------------------------------------------------------------------------------------
# * Use leave_one_image_out() for reporting decoding accuracy - it is the honest test of
#   category generalisation. leave_one_run_out() mixes repetition/novelty with run order
#   and lets test images leak into training (all 100 images repeat in every run).
# * epochs.metadata carries stim_id / category / run / coherence so you can build any
#   feature representation and still use the shared splits.
# * Filtering is MNE-native (notch_filter / filter) applied to ecog channels only.
