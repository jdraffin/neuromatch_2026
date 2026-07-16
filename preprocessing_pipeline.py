# %% Dependencies
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut

import detect_bad_channels as dbc

mne.set_log_level("ERROR")

# %% Parameters
# The pipeline works off either dataset; faces_basic is the primary one and the default.
# Pass dataset="faces_noise" (or a custom root=...) to switch. faces_basic has faceshouses
# only; faces_noise additionally has the fhnoisy task.
DATASETS = {
    "faces_basic": Path(r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_basic\data"),
    "faces_noise": Path(r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_noise\data"),
}
DATASET = "faces_basic"
data_root = DATASETS[DATASET]
subjects = dbc.DATASETS[DATASET]

notch = (60, 120, 180, 240)
bandpass = None
win_ms = (0, 400)

# %% Bad channels
# Curated bad-electrode lists (0-based, matching the ecogN channel names and the #N labels in
# faces_basic_all_subjects_full_traces.png). These are the confident calls from
# detect_bad_channels.py, cross-checked against visual inspection of the full-run traces.
# A channel bad in ANY run is dropped for the WHOLE subject (union), so every trial keeps the
# same channel set -- required by the fixed-feature decoders. Edit freely as review continues.
#
# Criteria per channel: DEAD = variance far below montage (invisible on the z-scored plot);
# SPIKY = epileptiform / regular-spiking (high kurtosis); NOISY+iso = hyper-variance AND
# decoupled from neighbours. See detect_bad_channels.py for thresholds.
MANUAL_BADS = {
    "aa": [0, 33, 41],                                  # 0,41 spiky; 33 dead
    "ap": [40],                                         # dead (also dead in faces_noise)
    "de": [13],                                         # dead (hidden by plot normalisation)
    "fp": [1, 2],                                       # hyper-variance + decoupled, all runs
    "ja": [37, 39, 51],                                 # regular spiking, all runs
    "jt": [31, 64, 65, 66, 67, 68, 69, 91, 96],         # decoupled hyper-noise/spiky bank
    # Candidates left IN pending eye-test (uncomment to drop):
    #   "aa": +[20]          # noisy in run 3 only
    #   "jt": +[50, 101]     # 50 high-variance but coupled ("dense"); 101 bad in run 1 only
    #   "rr": [2,3,4,5,33,34,35,36]  # elevated/epileptiform late in run 2 only (2 runs total)
    #   "zt": [3]            # mildly high-variance
    # ha: NOT channel drops -- whole-montage <1 Hz baseline drift; high-pass/detrend instead.
}

# Whole-montage dropouts to reject as time windows (not channel drops). Auto-detected too;
# listed here for documentation. jm: all channels rail ~256.2-261.3 s in run 3.


def _bad_channels(subject, dataset, root, task, bads):
    """Resolve the `bads` argument to a concrete list of 0-based channel indices."""
    if bads is None:
        return []
    if bads == "manual":
        return list(MANUAL_BADS.get(subject, []))
    if bads == "auto":
        return dbc.bad_channels(subject, dataset=dataset, task=task)
    return list(bads)                                   # explicit iterable of indices


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
def preprocess(subject, task="faceshouses", dataset=DATASET, root=None,
               notch=notch, bandpass=bandpass, win_ms=win_ms, bads="manual"):
    """Load, filter, mark/drop bad electrodes, and epoch one subject/task into MNE Epochs.

    bads : "manual" (curated MANUAL_BADS, default) | "auto" (detect_bad_channels) |
           None (keep all) | explicit iterable of 0-based channel indices.
           Dropped channels are recorded in raw.info["bads"] before being removed, so the
           set is auditable. Whole-montage dropout windows are marked as BAD_dropout
           annotations and any epoch overlapping them is rejected.
    """
    root = Path(root) if root is not None else DATASETS[dataset]
    raw, m = to_raw(root / subject / f"{subject}_{task}.mat", notch, bandpass)
    sf = raw.info["sfreq"]
    stim = np.ravel(m["stim"]).astype(int)

    # ---- bad channels: mark then drop (union across runs) ----
    bad_idx = _bad_channels(subject, dataset, root, task, bads)
    bad_names = [f"ecog{i}" for i in bad_idx if i < mne.pick_types(raw.info, ecog=True).size]
    raw.info["bads"] = bad_names
    if bad_names:
        raw.drop_channels(bad_names)

    # ---- whole-montage dropout windows -> annotations (rejected during epoching) ----
    for seg in dbc.global_rail_segments(np.asarray(m["data"], float), stim, sf):
        raw.set_annotations(raw.annotations + mne.Annotations(
            onset=seg["t0_s"], duration=max(seg["dur_s"], 1.0 / sf), description="BAD_dropout"))

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
                    preload=True, picks="ecog", metadata=meta,
                    reject_by_annotation=True)

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
