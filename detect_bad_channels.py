"""Bad-electrode detection for the Miller faces/houses ECoG datasets.

Why this exists
---------------
The shared full-trace figure (``faces_basic_all_subjects_full_traces.png``) is *robust
z-scored per channel and hard-clipped* for display. That normalisation is great for reading
the *character* of a trace (epileptiform spikes, regular spiking, drift) but it **destroys
amplitude information**: a dead / very-quiet channel is amplified until its noise floor looks
like a normal trace, and a hyper-variance channel is squashed/clipped so it looks ordinary.
So the eye-test alone cannot find amplitude-based problems (dead channels, hyper-noisy
channels). This module computes the metrics directly on the signal *in native units* so those
problems become visible.

All amplitude metrics are **relative to the subject's own montage** (robust z-score across
that subject/run's channels), because absolute scale varies wildly between subjects
(e.g. aa/ha/jt are ~1000x larger than the rest).

What it flags (per subject / per run / per channel)
---------------------------------------------------
  CLIP      rail_frac  : fraction of samples pinned at the int16 rails (+/-32767/8) or the
                         channel's own extreme -> hard saturation / clipping.
  DEAD      logvar_z   : log-variance robust-z far BELOW the montage -> dead / disconnected /
                         reference-quiet channel (INVISIBLE on the z-scored plot).
  NOISY     logvar_z   : log-variance robust-z far ABOVE the montage. NB high variance alone
                         is not necessarily bad (could be strong cortical signal); it is only
                         treated as bad when the channel is ALSO decoupled from its neighbours.
  SPIKY     kurtosis   : heavy-tailed / spiky -> epileptiform or spike artefact.
  ISOLATED  coupling_z : abnormally low correlation with the rest of the montage
                         (subject-relative) -> bad contact. Used as corroboration.

Two things that are NOT per-channel and are reported separately:
  * global-rail SEGMENTS: time windows where >50% of channels rail at once (e.g. jm run 3,
    256-261 s) -> reject the time window, do not drop channels.
  * global DRIFT: subjects whose whole montage is dominated by <1 Hz baseline wander
    (e.g. ha) -> high-pass / detrend, do not drop channels.

CLI
---
    python detect_bad_channels.py                     # faces_basic, all subjects -> CSV + heatmap
    python detect_bad_channels.py --dataset faces_noise
    python detect_bad_channels.py --subject jt        # one subject, verbose
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import iirnotch, filtfilt
from scipy.stats import kurtosis

DATASETS = {
    "faces_basic": ["aa", "ap", "ca", "de", "fp", "ha", "ja", "jm", "jt", "mv", "rn", "rr", "wc", "zt"],
    "faces_noise": ["ap", "ca", "ha", "ja", "mv", "wc", "zt"],
}
DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset"

NOTCH_FREQS = (60.0, 120.0, 180.0, 240.0)
NOTCH_Q = 30.0

# ---- thresholds (montage-relative unless noted) -----------------------------------------
T_RAIL = 0.002        # >0.2% of samples pinned at a rail -> CLIP
T_FLAT = 0.02         # >2% near-zero-derivative -> flat/dead segment
T_DEAD_Z = -3.5       # log-variance robust-z below this -> DEAD / quiet
T_NOISY_Z = 4.0       # log-variance robust-z above this -> high variance (mild)
T_NOISY_Z_HARD = 6.0  # ... strong
T_SPIKY = 12.0        # excess kurtosis (mild)
T_SPIKY_HARD = 25.0   # ... strong
T_ISO_Z = -2.5        # coupling robust-z below this -> ISOLATED from montage
T_DRIFT = 0.45        # subject-median 1-s-moving-average / total-std above this -> global drift


def _notch(x, sf):
    for f0 in NOTCH_FREQS:
        if f0 < sf / 2:
            b, a = iirnotch(f0, NOTCH_Q, sf)
            x = filtfilt(b, a, x, axis=0)
    return x


def _robust_z(v):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad == 0:
        mad = np.std(v) or 1.0
    return (v - med) / mad


def run_bounds(stim, min_gap_samples=2000):
    """Sample [start, end) of every run, using the stim==0 gaps between runs.

    Segments without any stimulus code (1..100) are dropped, so the short all-zero tail at
    the end of a recording is not counted as a run. Returns a list of (start, end).
    """
    stim = np.asarray(stim).ravel().astype(int)
    chg = np.where(np.diff(stim) != 0)[0] + 1
    starts = np.concatenate([[0], chg])
    ends = np.concatenate([chg, [len(stim)]])
    gaps = [(s, e) for s, e in zip(starts, ends)
            if stim[s] == 0 and (e - s) >= min_gap_samples and 1000 < s < len(stim) - 1000]
    bnds = [0] + [(s + e) // 2 for s, e in gaps] + [len(stim)]
    segs = [(bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)]
    return [(a, b) for a, b in segs if ((stim[a:b] >= 1) & (stim[a:b] <= 100)).any()]


def global_rail_segments(data, stim, sf):
    """Time windows (per run) where >50% of channels rail at the int16 limits simultaneously.

    These are whole-montage dropouts (e.g. jm run 3): reject the time window, do not drop
    channels. `data` is (time, channel). Returns a list of dicts with run/t0_s/t1_s/dur_s.
    """
    data = np.asarray(data, float)
    segs = []
    for ri, (a, b) in enumerate(run_bounds(stim), start=1):
        R = data[a:b]
        railed_t = (((np.abs(np.abs(R) - 32768) < 1.5) | (np.abs(np.abs(R) - 32767) < 1.5)).mean(1) > 0.5)
        if railed_t.any():
            idx = np.where(railed_t)[0]
            segs.append(dict(run=ri, t0_s=(a + idx.min()) / sf, t1_s=(a + idx.max()) / sf,
                             dur_s=(idx.max() - idx.min()) / sf))
    return segs


def _drift_ratio(x, w=1000):
    """Per-channel (1-s moving-average std) / (total std): fraction of amplitude in slow drift."""
    csum = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    mov = (csum[w:] - csum[:-w]) / w
    return mov.std(0) / (x.std(0) + 1e-9)


def _classify(m):
    """m: dict of scalar metrics for one channel/run -> (list_of_tags, drop_bool)."""
    tags, drop = [], False
    if m["rail_frac"] > T_RAIL:
        tags.append(f"CLIP({m['rail_frac']*100:.1f}%)"); drop = True
    if m["flat_frac"] > T_FLAT:
        tags.append(f"FLAT({m['flat_frac']*100:.1f}%)"); drop = True
    if m["logvar_z"] < T_DEAD_Z:
        tags.append(f"DEAD(z{m['logvar_z']:.1f})"); drop = True
    iso = m["coupling_z"] < T_ISO_Z
    if m["logvar_z"] > T_NOISY_Z_HARD:
        tags.append(f"NOISY(z{m['logvar_z']:.1f})")
        drop = drop or iso                    # hyper-variance + decoupled -> bad; else review
    elif m["logvar_z"] > T_NOISY_Z:
        tags.append(f"noisy(z{m['logvar_z']:.1f})")
    if m["kurtosis"] > T_SPIKY_HARD:
        tags.append(f"SPIKY(k{m['kurtosis']:.0f})"); drop = True
    elif m["kurtosis"] > T_SPIKY:
        tags.append(f"spiky(k{m['kurtosis']:.0f})")
    if iso:
        tags.append(f"iso(z{m['coupling_z']:.1f})")
    return tags, drop


def detect_subject(subject, dataset="faces_basic", root=DATA_ROOT, task="faceshouses"):
    """Return (rows, segments, drift) for one subject.

    rows     : list of per-(run, channel) dicts with metrics + tags + drop flag
    segments : list of dicts describing global-rail time windows (all-channel dropouts)
    drift    : dict with subject-median drift ratio and a global-drift bool
    """
    m = loadmat(Path(root) / dataset / "data" / subject / f"{subject}_{task}.mat")
    raw = np.asarray(m["data"], float)
    sf = float(np.ravel(m["srate"])[0])
    stim = np.ravel(m["stim"])
    filt = _notch(raw, sf)
    runs = run_bounds(stim)
    nchan = raw.shape[1]

    rows, segments, good_mask_full = [], [], np.ones(len(raw), bool)
    for ri, (a, b) in enumerate(runs, start=1):
        R, F = raw[a:b], filt[a:b]
        is_rail = (np.abs(np.abs(R) - 32768) < 1.5) | (np.abs(np.abs(R) - 32767) < 1.5)

        # global-rail segment (all-channel dropout) within this run: detect and EXCLUDE its
        # timepoints before scoring channels, so a whole-montage dropout (e.g. jm run 3) is
        # reported as a bad time window rather than condemning every channel.
        railed_t = is_rail.mean(1) > 0.5
        if railed_t.any():
            idx = np.where(railed_t)[0]
            segments.append(dict(run=ri, t0_s=(a + idx.min()) / sf, t1_s=(a + idx.max()) / sf,
                                 dur_s=(idx.max() - idx.min()) / sf, n_samples=int(railed_t.sum())))
            # dilate the excluded window by ~0.5 s each side so the zero-phase notch filter's
            # ringing at the dropout edges does not masquerade as per-channel spikes.
            margin = int(0.5 * sf)
            railed_t = np.convolve(railed_t, np.ones(2 * margin + 1, bool), "same") > 0
            good_mask_full[a:b][railed_t] = False
        good = ~railed_t
        R, F, is_rail = R[good], F[good], is_rail[good]

        chmax, chmin = R.max(0), R.min(0)
        sat = ((np.abs(R - chmax) < 1e-6) | (np.abs(R - chmin) < 1e-6)).mean(0)
        rail16 = is_rail.mean(0)
        rail = np.maximum(sat * (sat > 0.02), rail16)   # own-rail only if very sustained
        flat = (np.abs(np.diff(R, axis=0)) < 1e-9).mean(0)
        logvar = np.log10(np.var(F, axis=0) + 1e-12)
        logvar_z = _robust_z(logvar)
        kurt = kurtosis(F, axis=0, fisher=True)
        C = np.corrcoef(F[::5].T); np.fill_diagonal(C, np.nan)
        coupling = np.nanmedian(np.abs(C), axis=1)
        coupling_z = _robust_z(coupling)

        for ch in range(nchan):
            met = dict(rail_frac=float(rail[ch]), flat_frac=float(flat[ch]),
                       logvar_z=float(logvar_z[ch]), kurtosis=float(kurt[ch]),
                       coupling_z=float(coupling_z[ch]))
            tags, drop = _classify(met)
            rows.append(dict(subject=subject, run=ri, ch=ch, **met,
                             tags=" ".join(tags), is_bad=bool(drop)))

    if runs:
        span = slice(runs[0][0], runs[-1][1])
        dr = _drift_ratio(filt[span][good_mask_full[span]])
    else:
        dr = np.array([0.0])
    drift = dict(median=float(np.median(dr)), global_drift=bool(np.median(dr) > T_DRIFT))
    return rows, segments, drift


def bad_channels(subject, dataset="faces_basic", root=DATA_ROOT, task="faceshouses"):
    """Union of auto-dropped channels across runs (0-based) for one subject.

    Union = "bad in ANY run -> bad for the whole subject", giving a constant channel set
    across all trials (required by fixed-feature decoders).
    """
    rows, _, _ = detect_subject(subject, dataset, root, task)
    return sorted({r["ch"] for r in rows if r["is_bad"]})


def detect_all(dataset="faces_basic", root=DATA_ROOT, task="faceshouses"):
    import pandas as pd
    all_rows, meta = [], {}
    for s in DATASETS[dataset]:
        rows, segments, drift = detect_subject(s, dataset, root, task)
        all_rows += rows
        meta[s] = dict(segments=segments, drift=drift)
    return pd.DataFrame(all_rows), meta


# --------------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="faces_basic", choices=list(DATASETS))
    ap.add_argument("--subject", default=None, help="one subject (default: all)")
    ap.add_argument("--out", default=None, help="CSV path (default: <dataset>_bad_channels.csv here)")
    args = ap.parse_args()

    subs = [args.subject] if args.subject else DATASETS[args.dataset]
    frames, meta = [], {}
    for s in subs:
        rows, segments, drift = detect_subject(s, args.dataset)
        frames += rows
        meta[s] = dict(segments=segments, drift=drift)
    df = pd.DataFrame(frames)

    out = args.out or str(Path(__file__).resolve().parent / f"{args.dataset}_bad_channels.csv")
    df[df.is_bad].to_csv(out, index=False)
    print(f"saved auto-drop rows -> {out}\n")

    for s in subs:
        sub = df[(df.subject == s) & (df.is_bad)]
        drop_union = sorted(sub.ch.unique().tolist())
        segs = meta[s]["segments"]; dr = meta[s]["drift"]
        head = f"{s}: drop(union)={drop_union}" if drop_union else f"{s}: (no channels auto-dropped)"
        if dr["global_drift"]:
            head += f"  [GLOBAL DRIFT median={dr['median']:.2f} -> high-pass, don't drop]"
        print(head)
        for seg in segs:
            print(f"    !! global-rail segment run{seg['run']} {seg['t0_s']:.1f}-{seg['t1_s']:.1f}s "
                  f"({seg['dur_s']:.1f}s) -> reject time window, not channels")
        for _, r in sub.sort_values(["run", "ch"]).iterrows():
            print(f"    run{r['run']} #{int(r['ch']):>3}  {r['tags']}")
