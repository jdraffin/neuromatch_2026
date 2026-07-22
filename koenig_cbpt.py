"""Port of Konig et al. (2024) cluster-based permutation testing with (G)LMEs.

SOURCE OF RECORD: reference_koenig2024_matlab/clusterBasedGlmeERD.m (Herman-Darrow Lab,
CC BY-SA 4.0), accompanying:

  Konig, Safo, Miller, Herman & Darrow (2024). "Flexible multi-step hypothesis testing of
  human ECoG data using cluster-based permutation tests with GLMEs." NeuroImage 290:120557.

This is a port, not a reimplementation-from-the-paper. Function/variable names below
mirror the MATLAB so the two can be diffed. Line references are to clusterBasedGlmeERD.m.

Fidelity notes (deliberate, and the only places this departs from the .m):

  * fitlme -> statsmodels MixedLM. MATLAB's fitlme defaults to ML; we pass reml=False to
    match. MATLAB reports `tStat`; statsmodels' `.tvalues` are Wald z. For the n used here
    (>=200 trials) these coincide to ~3 decimals, but the cluster statistic is therefore
    sum(z^2) not sum(t^2).
  * MATLAB `Coefficients` always drops the intercept via `(2:end)`. We drop the term named
    'Intercept' explicitly, which is the same set for the formulas used here.
  * numFixedEffects = NumCoefficients-1 counts COEFFICIENTS, not model terms, so a
    3-level factor contributes 2 and an interaction contributes its own. We replicate that
    by counting fitted non-intercept coefficients rather than formula terms (.m line 232).
  * On a failed fit MATLAB sets p=1 ("not significant", .m line 280); we do the same.
  * Ties for the largest cluster: MATLAB `find(x == max(x))` returns all tied indices and
    then indexes with the vector (.m line 414-417), which silently takes a wider span. We
    take the first, which is what that code does whenever the max is unique (always, in
    practice, for continuous statistics).

NOT ported: the 'burstRate' logistic/Poisson branch (we have no burst data) and
clusterBasedGlmeERD2D.m. `cluster_definition='uniqueCombos'` IS ported.

Only ONE random intercept is supported, passed as `groups`. That covers the paper's
per-channel model and its first three group models, including the one their example script
selects as best ('erd ~ eventType + novelRepeat + (1|allChannelNumbers)'). Their hierarchical
group model and their alternative models need two crossed or nested random intercepts
(e.g. '+ (1|allChannelNumbers) + (1|eventValue)'); those would need a vc_formula path here
and are NOT available.

VALIDATION (scr_331_validate_koenig_port.py, run 2026-07-22)
------------------------------------------------------------
An earlier version of this docstring claimed the port was "verified equivalent on the
reference data (see test at bottom)". There was no test at the bottom, and
reference_koenig2024_matlab/ shipped no data, so nothing supported that claim. It has now
been checked properly.

The authors' example file (exampleGlmeFaceHouseAnalysisData.mat, Zenodo 10.5281/zenodo.7703148,
CC BY-4.0) is derived from Miller's faces_basic library, which we already hold. Its event
tables are MATLAB `table` objects that neither scipy.io nor pymatreader can decode, so the
source channel was identified by correlation instead and its event table regenerated from
that subject's own stim vector: selectedBBData{1} is subject jm, ecog12, all 300 trials in
acquisition order (trial-mean r = 1.000 against dmdsdm.miller_broadband; full trial x time
r = 0.984, the residual being their per-channel z-scoring).

Run at their own published parameters -- 'erd ~ eventType + novelRepeat + (1|eventValue)',
alphaIndividual 0.04, minClusterSize 55 ms, numShuffs 1000, window -200..600 ms -- this port
returns significantGlmeFit == 1 with exactly two significant clusters, +144..+382 ms
(sum t^2 = 35.4) and +461..+523 ms (sum t^2 = 8.6). That matches the result
exampleFaceHouseAnlaysisBB.m expects: it titles the panel "Significant Channel" and indexes
sigGlmes{1}{1} and sigGlmes{1}{2}, i.e. two significant clusters.

Caveat on how strong that is: the authors publish no table of cluster boundaries or p-values,
and MATLAB is not available here, so the agreement checked is the count and significance of
clusters, not their exact edges. A boundary difference of a few ms from optimizer or
z-vs-t differences would not be detected.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def findgaps(input_ind):
    """Port of findgaps.m. Contiguous runs in a sorted index vector -> (starts, ends)."""
    input_ind = np.asarray(input_ind, dtype=int)
    if input_ind.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    gaps = np.where(np.abs(np.diff(input_ind)) > 1)[0]
    if gaps.size == 0:
        return np.array([input_ind[0]]), np.array([input_ind[-1]])
    starts, ends = [], []
    for g in range(len(gaps) + 1):
        if g == 0:
            temp = input_ind[: gaps[g] + 1]
        elif g == len(gaps):
            temp = input_ind[gaps[g - 1] + 1:]
        else:
            temp = input_ind[gaps[g - 1] + 1: gaps[g] + 1]
        starts.append(temp[0]); ends.append(temp[-1])
    return np.array(starts), np.array(ends)


_OPTIMIZERS = (None, "bfgs", "powell")


def _fit(df, formula, groups):
    """One fitlme() call. Returns (coef_names, pvalues, tstats, betas, se) sans intercept.

    Optimizer cascade: in this design `category` is a deterministic function of `imageID`
    (codes 1-50 house, 51-100 face), so the random intercept is perfectly nested within
    the category contrast. MATLAB's fitlme handles that; statsmodels' 'lbfgs' returns a
    singular Hessian on it. The default optimizer, 'bfgs' and 'powell' all converge and
    agree to 3 decimals, so we try them in turn rather than declaring the fit failed.
    """
    model = smf.mixedlm(formula, df, groups=df[groups])
    last = None
    for opt in _OPTIMIZERS:
        try:
            r = model.fit(reml=False) if opt is None else model.fit(reml=False, method=opt)
            break
        except Exception as e:                      # singular Hessian etc -> next optimizer
            last = e
    else:
        raise last
    keep = [c for c in r.params.index
            if "Intercept" not in c and c != "Group Var" and not c.endswith("Var")]
    return (keep,
            np.array([r.pvalues[c] for c in keep]),
            np.array([r.tvalues[c] for c in keep]),
            np.array([r.params[c] for c in keep]),
            np.array([r.bse[c] for c in keep]))


def cluster_based_glme_erd(glme_formula, erd, event_table, groups, time_window,
                           min_cluster_size, num_shuffs=1000, alpha_value=0.05,
                           time_window_of_interest=None,
                           cluster_definition="allTogether", rng=None):
    """Port of clusterBasedGlmeERD.m for a single channel of normally-distributed ERD.

    Parameters mirror the MATLAB argument list.

    glme_formula  : Wilkinson-ish formula for the FIXED part, dependent variable 'erd'
                    (e.g. 'erd ~ eventType + novelRepeat'). The random intercept is given
                    separately via `groups`, since statsmodels takes it as an argument
                    rather than in the formula.
    erd           : (n_trials, n_time) dependent variable.
    event_table   : DataFrame, n_trials rows, holding fixed- and random-effect columns.
    groups        : column name in event_table for the random intercept, i.e. the
                    '(1|x)' term.
    time_window   : (n_time,) timestamps in ms, evenly spaced.
    min_cluster_size : minimum cluster length in MILLISECONDS (paper's face/house analysis
                    used 55 ms; see exampleFaceHouseAnlaysisBB.m).

    Returns a dict mirroring `glmeClusterStats`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    erd = np.asarray(erd, dtype=float)
    time_window = np.asarray(time_window, dtype=float)

    # .m lines 134-137: sampling rate and window-of-interest indices
    sampling_rate = 1e3 / np.mean(np.diff(time_window))
    min_cluster_samples = min_cluster_size / 1e3 * sampling_rate
    if time_window_of_interest is None:
        i0, i1 = 0, len(time_window) - 1
    else:
        i0 = int(np.argmin(np.abs(time_window - time_window_of_interest[0])))
        i1 = int(np.argmin(np.abs(time_window - time_window_of_interest[1])))
    erd = erd[:, i0:i1 + 1]
    tw = time_window[i0:i1 + 1]
    n_time = erd.shape[1]

    df = event_table.reset_index(drop=True).copy()

    # ---- .m lines 219-310: fit the LME at every time point
    names, P, T, B, SE = None, None, None, None, None
    for k in range(n_time):
        d = df.copy(); d["erd"] = erd[:, k]
        try:
            nm, p, t, b, se = _fit(d, glme_formula, groups)
        except Exception:                                   # .m line 275-281
            if names is None:
                raise
            nm = names
            p = np.ones(len(names)); t = np.full(len(names), np.nan)
            b = np.full(len(names), np.nan); se = np.full(len(names), np.nan)
        if names is None:
            names = nm
            P = np.full((n_time, len(names)), np.nan)
            T = np.full((n_time, len(names)), np.nan)
            B = np.full((n_time, len(names)), np.nan)
            SE = np.full((n_time, len(names)), np.nan)
        P[k], T[k], B[k], SE[k] = p, t, b, se
    num_fixed_effects = len(names)                          # .m line 232 (coefficients)

    # ---- .m lines 325-378: define clusters
    if cluster_definition.lower() == "alltogether":
        thr = alpha_value if num_fixed_effects == 1 else alpha_value / num_fixed_effects
        combined = np.any(P < thr, axis=1)                  # .m lines 362-366
        cs, ce = findgaps(np.where(combined)[0])
    elif cluster_definition.lower() == "uniquecombos":
        sig = P < alpha_value / num_fixed_effects           # .m lines 329-345
        combos = np.array([int("".join("1" if v else "0" for v in row), 2) for row in sig])
        cs, ce = [], []
        for c in np.unique(combos):
            if c == 0:
                continue
            s, e = findgaps(np.where(combos == c)[0])
            cs.extend(s.tolist()); ce.extend(e.tolist())
        order = np.argsort(cs)                              # .m lines 347-350 (resort)
        cs, ce = np.array(cs)[order], np.array(ce)[order]
    else:
        raise ValueError("cluster definition type unknown!")

    if len(cs):                                             # .m lines 370-374 (too small)
        dur = ce - cs + 1
        big = dur >= min_cluster_samples
        cs, ce = cs[big], ce[big]

    out = dict(numFixedEffects=num_fixed_effects, fixedEffectNames=names,
               betaWeightsTimePoints=B, sigIndividualTimePoints=P,
               tStatIndividualTimePoints=T, seIndividualTimePoints=SE,
               timeWindow=tw, clusterStart=cs, clusterEnd=ce,
               sumSquaredTstats=np.array([]), shuffTstats=np.array([]),
               sigClusters=np.array([]), significantGlmeFit=0,
               significantGlmeTimes=np.zeros(n_time, dtype=int))
    if not len(cs):                                         # .m line 483
        return out

    # ---- .m lines 381-410: cluster-level statistic = sum of squared t, intercept dropped
    def _cluster_stat(vals):
        d = df.copy(); d["erd"] = vals
        try:
            _, _, t, _, _ = _fit(d, glme_formula, groups)
            return float(np.nansum(t ** 2))
        except Exception:                                   # .m line 407-409
            return np.nan

    sum_sq = np.array([_cluster_stat(erd[:, a:b + 1].mean(axis=1)) for a, b in zip(cs, ce)])

    # ---- .m lines 412-457: permute the LARGEST cluster only (Manly)
    biggest = int(np.nanargmax(sum_sq))
    biggest_data = erd[:, cs[biggest]:ce[biggest] + 1].mean(axis=1)
    n_trials = len(biggest_data)
    shuff = np.full(num_shuffs, np.nan)
    for s in range(num_shuffs):
        shuff[s] = _cluster_stat(biggest_data[rng.permutation(n_trials)])

    # ---- .m lines 459-467: count-based threshold, NOT a percentile
    sig_clusters = np.array([np.nansum(v > shuff) > (1 - alpha_value) * num_shuffs
                             for v in sum_sq], dtype=int)

    sig_times = np.zeros(n_time, dtype=int)                 # .m lines 471-478
    for i, (a, b) in enumerate(zip(cs, ce)):
        if sig_clusters[i]:
            sig_times[a:b + 1] = i + 1
    out.update(sumSquaredTstats=sum_sq, shuffTstats=shuff, sigClusters=sig_clusters,
               significantGlmeFit=int(np.any(sig_clusters == 1)),
               significantGlmeTimes=sig_times)
    return out
