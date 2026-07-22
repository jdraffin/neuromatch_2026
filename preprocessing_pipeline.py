# %% Dependencies
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneGroupOut

mne.set_log_level("ERROR")

# %% Parameters
# The pipeline works off either dataset; faces_basic is the primary one and the default.
# Pass dataset="faces_noise" (or a custom root=...) to switch. faces_basic has faceshouses
# only; faces_noise additionally has the fhnoisy task.
DATASETS = {
    "faces_basic": Path(r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_basic\data"),
    "faces_noise": Path(r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_noise\data"),
}
# THE CANONICAL COHORT. The seven subjects of Miller et al. 2016, PLoS Comput Biol
# 12(1):e1004660, in paper subject order 1-7. This is the default cohort for ALL analyses:
# it is the published sample, every one of these subjects appears in the paper's figures, and
# results on it are directly comparable to the literature.
#
# Source of record: dataset/faces_basic/README_faces_basic_dataset_notes.docx - Miller's own
# data-release note - which states "The corresponding subject number for each patient from the
# manuscript is: ja 1, ca 2, mv 3, wc 4, de 5, zt 6, fp 7".
# miller_subject_mapping_and_exclusions.xlsx agrees on membership; it numbers mv/wc/zt
# differently and its own Notes column flags that uncertainty, so prefer the README.
#
# BEWARE two other 7-element sets in this project that are NOT this one:
#   * SUBJECTS["faces_noise"] below (ap ca ha ja mv wc zt) - the fhnoisy cohort, which differs
#     from MILLER7 on de/fp vs ap/ha. dmd_python/pair_common.py:SUBS is a copy of it.
#   * SUBJECTS_EXTRA below (aa ap ha jm jt rn rr) - faces_basic subjects NOT in the paper.
MILLER7 = ["ja", "ca", "mv", "wc", "de", "zt", "fp"]

# The seven faces_basic subjects that are not in the 2016 paper. Kept for robustness checks
# and for anything that wants the full folder; not part of the default cohort.
SUBJECTS_EXTRA = ["aa", "ap", "ha", "jm", "jt", "rn", "rr"]
SUBJECTS_ALL14 = sorted(MILLER7 + SUBJECTS_EXTRA)

SUBJECTS = {
    "faces_basic": MILLER7,          # default = the published cohort, NOT all 14
    "faces_noise": ["ap", "ca", "ha", "ja", "mv", "wc", "zt"],
}
COHORTS = {"miller7": MILLER7, "extra7": SUBJECTS_EXTRA, "all14": SUBJECTS_ALL14}
DATASET = "faces_basic"
data_root = DATASETS[DATASET]
subjects = SUBJECTS[DATASET]

notch = (60, 120, 180, 240)
bandpass = None
win_ms = (0, 400)

# %% Exclusions (edit by hand)
# Electrodes to exclude, at subject x electrode x run x trial granularity. Nested dict:
#     { subject: { electrode : where } }
# electrode = 0-based channel index (matches ecogN and the #N labels in the trace figures).
# where     = one of three forms, coarse to fine:
#     "all"              -> excluded everywhere.
#     [0, 2]             -> excluded in whole runs 0 and 2 (0-based run numbers).
#     {2: [(34, 53)]}    -> excluded only in run 2, trials 34-53. Trials are run-local and
#                           0-based (they restart each run), matching the trial ticks drawn
#                           above each subject block by plot_full_traces.py, so a number read
#                           off that figure can be pasted straight in. Each entry is a single
#                           index or an inclusive (start, end) range. A run may also map to
#                           "all" for the whole run.
#
# A range may open with "start" and close with "end", meaning the run's own boundary rather
# than merely its first/last trial: ("start", 6) also covers the interval BEFORE trial 0, and
# (34, "end") also covers the interval AFTER the last trial. For epoching that changes nothing
# (an epoch only exists at a trial), but it is the honest record of where the electrode is
# actually unusable, it is what the trace figure shades, and it matters if the continuous data
# either side of the trials is ever analysed. Prefer it to a bare 0 / last-trial index.
#
#   EXCLUDE_CHANNELS = {
#       "jt": {64: "all", 65: [1]},                 # 64 everywhere; 65 in the whole of run 1.
#       "aa": {41: {0: [(0, 6)], 2: [(23, 37)]}},   # 41 only in those trial ranges of runs 0 and 2.
#   }
#
# How it is applied (epochs are a fixed trials x channels x time array, so a channel cannot be
# present for some trials and absent for others):
#   * excluded on EVERY kept trial -> the channel is DROPPED for the whole subject.
#   * excluded on only SOME trials -> the channel is KEPT, but those electrode x trial cells
#     are NaN'd (PARTIAL_ACTION="nan", the default) or left intact ("mask"); either way each
#     trial lists its excluded channels in epochs.metadata["exclude_channels"].
EXCLUDE_CHANNELS = {
    # Read off faces_basic_all_subjects_full_traces_{raw,zscored}.png by hand.
    "aa": {
        0:  {2: [(34, 53)]},
        7:  {2: [(34, 38)]},
        20: {2: [(34, "end")]},
        25: {2: [(33, 35)]},
        41: {0: [("start", 6)], 1: [("start", 6), (92, "end")], 2: [("start", 5), (23, 37)]},
    },
    "ja": {37: "all", 38: "all"},
    "jt": {64: "all", 65: "all", 66: "all", 67: "all", 68: "all", 69: "all", 91: "all", 96: "all"},
    "wc": {26: "all", 28: "all"},
}
PARTIAL_ACTION = "nan"        # "nan" = NaN partial-run excluded cells; "mask" = annotate only

# Whole runs to exclude, per subject: 0-based run numbers. Every trial in a listed run is
# dropped (use for a run ruined across all electrodes, not a single bad electrode - that is
# EXCLUDE_CHANNELS).
EXCLUDE_RUNS = {
    # "rr": [1],      # drop the 2nd run
}

# Individual trials to exclude, per subject and run: { subject: { run : trials } }
# run    = 0-based run number (matches the "run" column in epochs.metadata).
# trials = 0-based trial numbers *within that run* (run-local, so the first trial of every
#          run is 0). Each entry is either a single index or an inclusive (start, end) range.
# The listed trials are dropped entirely, like EXCLUDE_RUNS but at finer granularity (use for
# a stretch of a run ruined across all electrodes, e.g. the subject moving or an amp glitch).
#
#   EXCLUDE_TRIALS = {
#       "rr": {0: [(10, 19)], 1: [3, 7, (40, 44)]},   # run 0: trials 10-19; run 1: 3, 7 and 40-44.
#   }
EXCLUDE_TRIALS = {
    "jm": {2: [(87, 95)]},
}


def _expand_trials(trials, last=None):
    """Expand a trial spec (ints and inclusive (start, end) ranges) into a set of indices.

    A range may open with "start" (the run's first trial) and close with "end" (its last);
    `last` supplies that final index. For epochs those sentinels are just trial 0 and the last
    trial, but they additionally mean "reach the run's own boundary" - see the note on
    EXCLUDE_CHANNELS about the pre-first-trial and post-last-trial intervals.
    """
    out = set()
    for item in trials:
        if isinstance(item, (tuple, list)):
            start, end = item
            start = 0 if start == "start" else int(start)
            if end == "end":
                if last is None:
                    raise ValueError('"end" used where the run length is unknown')
                end = last
            out.update(range(start, int(end) + 1))
        else:
            out.add(int(item))

    return out


def _channel_trial_mask(where, runs, trial_in_run, run_last):
    """Bool mask over trials: is this electrode excluded on each trial?

    `where` is an EXCLUDE_CHANNELS value ("all" | [runs] | {run: trials}).
    """
    if isinstance(where, str):
        if where != "all":
            raise ValueError(f"bad EXCLUDE_CHANNELS value {where!r}")
        return np.ones(len(runs), dtype=bool)
    if isinstance(where, dict):
        mask = np.zeros(len(runs), dtype=bool)
        for run, trials in where.items():
            in_run = runs == run
            if isinstance(trials, str) and trials == "all":
                mask |= in_run
            else:
                sel = _expand_trials(trials, run_last.get(run))
                mask |= in_run & np.isin(trial_in_run, list(sel))
        return mask

    return np.isin(runs, list(where))            # list of whole runs


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
        # 3rd-order Butterworth, applied forward-backward (phase="zero" -> zero-phase filtfilt).
        # MNE's IIR path takes one stop-band at a time, so the harmonics are looped over.
        for f in notch:
            if f < sf / 2:
                raw.notch_filter([f], picks=picks, method="iir",
                                 iir_params=dict(order=3, ftype="butter", output="sos"),
                                 phase="zero")
    if bandpass:
        raw.filter(bandpass[0], bandpass[1], picks=picks)

    return raw, m


# %% Run detection
# Subjects whose runs were concatenated WITHOUT the usual inter-run silence, so the
# gap-based detector below sees a single run. The faceshouses design is 3 runs x 100
# trials for every subject (verified: all 7 Miller subjects have 300 onsets, and the six
# detectable ones split exactly 100/100/100 at two ~9080-sample stim==0 gaps). `ha` has
# only the trailing gap, so its boundaries are recovered from the known trial count.
# value = number of equal, contiguous runs to split the onsets into.
RUN_OVERRIDES = {"ha": 3}


def _run_index(stim, onsets, min_gap_samples=2000, subject=None):
    """Assign each stimulus onset to a run using the stim==0 gaps between runs.

    `subject` enables the RUN_OVERRIDES fallback: if the gap detector finds no run
    boundary for a subject known to have concatenated runs, split the onsets into equal
    contiguous blocks in acquisition order. Without `subject` the behaviour is unchanged.
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

    n_forced = RUN_OVERRIDES.get(subject)
    if n_forced and len(np.unique(run)) == 1 and len(onsets) % n_forced == 0:
        per = len(onsets) // n_forced           # onsets are in acquisition order
        run = np.arange(len(onsets), dtype=int) // per
    return run


# %% Epoching
def preprocess(subject, task="faceshouses", dataset=DATASET, root=None,
               notch=notch, bandpass=bandpass, win_ms=win_ms,
               exclude_channels=None, exclude_runs=None, exclude_trials=None, partial=None):
    """Load, filter, and epoch one subject/task into MNE Epochs with metadata.

    exclude_channels : {electrode: runs} for this subject (electrode 0-based; runs a list of
                   0-based run numbers or "all"). None -> EXCLUDE_CHANNELS.get(subject, {}).
                   Excluded-in-all-runs electrodes are dropped; excluded-in-some-runs
                   electrodes are kept and those trials NaN'd/annotated (see `partial`).
    exclude_runs : 0-based run numbers whose trials to drop entirely. None -> EXCLUDE_RUNS.
    exclude_trials : {run: trials} for this subject, where trials are run-local 0-based trial
                   numbers given as single indices and/or inclusive (start, end) ranges.
                   Those trials are dropped. None -> EXCLUDE_TRIALS.get(subject, {}).
    partial      : "nan" | "mask" for excluded-in-some-runs electrodes. None -> PARTIAL_ACTION.

    epochs.metadata gains an "exclude_channels" column: per trial, the channel names flagged
    for exclusion in that trial's run.
    """
    root = Path(root) if root is not None else DATASETS[dataset]
    raw, m = to_raw(root / subject / f"{subject}_{task}.mat", notch, bandpass)
    sf = raw.info["sfreq"]
    stim = np.ravel(m["stim"]).astype(int)
    n_ecog = len(mne.pick_types(raw.info, ecog=True))
    spec = EXCLUDE_CHANNELS.get(subject, {}) if exclude_channels is None else exclude_channels
    partial = PARTIAL_ACTION if partial is None else partial

    events = mne.find_events(raw, stim_channel="STIM", consecutive=True)

    if task == "faceshouses":
        events = events[(events[:, 2] >= 1) & (events[:, 2] <= 100)]
        codes = events[:, 2]
        meta = pd.DataFrame({
            "stim_id": codes,
            "category": np.where(codes <= 50, "house", "face"),
            "run": _run_index(stim, events[:, 0], subject=subject),
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

    # ---- exclude whole runs, then individual trials within a run (drop their trials) ----
    # trial index within its run, before any dropping, so hand-listed numbers stay stable
    trial_in_run = meta.groupby("run").cumcount()
    run_last = trial_in_run.groupby(meta["run"]).max().to_dict()   # last trial index of each run

    drop_runs = EXCLUDE_RUNS.get(subject, []) if exclude_runs is None else exclude_runs
    keep = ~meta["run"].isin(drop_runs) if drop_runs else pd.Series(True, index=meta.index)

    drop_trials = EXCLUDE_TRIALS.get(subject, {}) if exclude_trials is None else exclude_trials
    for run, trials in drop_trials.items():
        sel = _expand_trials(trials, run_last.get(run))
        keep &= ~((meta["run"] == run) & trial_in_run.isin(sel))

    events = events[keep.to_numpy()]
    meta = meta[keep].reset_index(drop=True)
    trial_in_run = trial_in_run[keep].to_numpy()          # run-local ids of the surviving trials

    # ---- resolve excluded electrodes into whole-channel drops vs per-trial NaNs ----
    runs_arr = meta["run"].to_numpy()
    excl_mask = {}                                        # {electrode: bool mask over trials}
    for elec, where in spec.items():
        if not (0 <= elec < n_ecog):
            continue
        mask = _channel_trial_mask(where, runs_arr, trial_in_run, run_last)
        if mask.any():
            excl_mask[elec] = mask

    # excluded on every surviving trial -> the channel is useless, drop it outright
    drop_names = [f"ecog{e}" for e, mask in excl_mask.items() if mask.all()]
    raw.info["bads"] = drop_names
    if drop_names:
        raw.drop_channels(drop_names)
    partial_cells = {e: mask for e, mask in excl_mask.items() if not mask.all()}

    # per-trial list of channels excluded on that trial
    excl_per_trial = [[] for _ in range(len(meta))]
    for elec, mask in partial_cells.items():
        for i in np.flatnonzero(mask):
            excl_per_trial[i].append(f"ecog{elec}")
    meta["exclude_channels"] = excl_per_trial

    tmin, tmax = win_ms[0] / 1000.0, win_ms[1] / 1000.0 - 1.0 / sf
    ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None,
                    preload=True, picks="ecog", metadata=meta)

    # NaN the partially-excluded electrode x trial cells
    if partial == "nan" and partial_cells:
        for elec, mask in partial_cells.items():
            name = f"ecog{elec}"
            if name in ep.ch_names:
                ep._data[mask, ep.ch_names.index(name), :] = np.nan

    return ep


# %% Splits
def _y(epochs):
    return (epochs.metadata["category"].to_numpy() == "face").astype(int)


# Fixed 5-fold image-identity CV split, stratified by category (10 houses and 10 faces per fold).
# Hardcoded from fixed_image_identity_cv_folds.csv (Diana's file)
IMAGE_CV_FOLDS = {
    1: [5, 10, 13, 22, 23, 27, 31, 32, 33, 46, 54, 55, 61, 63, 68, 78, 81, 86, 95, 100],
    2: [1, 9, 11, 19, 21, 25, 29, 39, 43, 45, 51, 62, 69, 70, 73, 74, 79, 84, 93, 99],
    3: [3, 14, 15, 16, 20, 24, 37, 38, 48, 49, 56, 57, 58, 64, 65, 77, 87, 90, 96, 97],
    4: [2, 4, 6, 18, 26, 28, 41, 42, 44, 47, 52, 53, 60, 67, 80, 82, 85, 88, 89, 92],
    5: [7, 8, 12, 17, 30, 34, 35, 36, 40, 50, 59, 66, 71, 72, 75, 76, 83, 91, 94, 98],
}
_STIM_TO_FOLD = {s: f for f, stims in IMAGE_CV_FOLDS.items() for s in stims}


def leave_one_image_out(epochs, n_splits=5, seed=0):
    if n_splits != 5:
        raise ValueError(f"IMAGE_CV_FOLDS defines exactly 5 fixed folds; got n_splits={n_splits}")
    stim = epochs.metadata["stim_id"].to_numpy()
    missing = sorted(set(stim.tolist()) - _STIM_TO_FOLD.keys())
    if missing:
        raise ValueError(
            f"stim_id(s) {missing} are not in the fixed faceshouses image-fold map; "
            "leave_one_image_out is defined only for the 100 faceshouses images"
        )
    fold_of = np.array([_STIM_TO_FOLD[s] for s in stim])
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(1, 6)]


def leave_one_run_out(epochs):
    """Group by run for cross-run stability analyses (faceshouses only)."""
    groups = epochs.metadata["run"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("only one run present (e.g. subject ha) - leave-one-run-out N/A")
    Xd = np.zeros((len(groups), 1))

    return list(LeaveOneGroupOut().split(Xd, groups=groups))
