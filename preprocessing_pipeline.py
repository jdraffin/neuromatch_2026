# %% Dependencies
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut

mne.set_log_level("ERROR")

# %% Parameters
data_root = r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_basic\data"
subjects = ["aa", "ap", "ca", "de", "fp", "ha", "ja", "jm", "jt", "mv", "rn", "rr", "wc", "zt"]

notch = (60, 120, 180, 240)
bandpass = None
win_ms = (0, 400)


# %% Raw and filter
def to_raw(mat_path, notch=notch, bandpass=bandpass):
    """Load a .mat file into an MNE RawArray and return the mat dict"""
    m = loadmat(mat_path)
    sf = float(np.ravel(m["srate"])[0])
    data = np.asarray(m["data"], dtype=float).T
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


# %% Run detection
def _run_index(stim, onsets, min_gap_samples=2000):
    """Assign each stimulus onset to a run using the stim==0 gaps between runs"""
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


# %% Epoching
def preprocess(subject, task="faceshouses", root=data_root,
               notch=notch, bandpass=bandpass, win_ms=win_ms):
    """Load, filter, and epoch one subject/task into MNE epochs with metadata"""
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
        events = events[events[:, 2] >= 1]
        tr_fh = np.ravel(m["tr_fh"]).astype(int)
        tr_coh = np.ravel(m["tr_coh"]).astype(int)
        n = min(len(events), len(tr_fh))
        events, tr_fh, tr_coh = events[:n], tr_fh[:n], tr_coh[:n]
        meta = pd.DataFrame({
            "stim_id": np.arange(1, n + 1),
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


# %% Splits
def _y(epochs):
    return (epochs.metadata["category"].to_numpy() == "face").astype(int)


def leave_one_image_out(epochs, n_splits=5, seed=0):
    """Group by stimulus id, stratified by category, holding out all repeats of test images"""
    y = _y(epochs)
    groups = epochs.metadata["stim_id"].to_numpy()
    Xd = np.zeros((len(y), 1))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(Xd, y, groups))


def leave_one_run_out(epochs):
    """Group by run for cross-run stability analyses (faceshouses only)."""
    groups = epochs.metadata["run"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("only one run present (e.g. subject ha) - leave-one-run-out N/A")
    Xd = np.zeros((len(groups), 1))

    return list(LeaveOneGroupOut().split(Xd, groups=groups))
