"""LC-MS run-to-run retention-time alignment.

LC retention times drift between samples (column ageing, gradient
reproducibility, lab temperature, etc.).  Even a 30-second drift across
36 mzML files is enough to push the same peptide into a different
cycle's MS2 window, increasing missing values and hurting site-level
quantification.  This module aligns every run's RTs onto a common
reference using **LOWESS** (locally weighted scatterplot smoothing),
which is the same approach used by MaxQuant, Skyline and FragPipe-DIA.

Algorithm
---------
1. Pick a *reference run* — by default the run with the largest number of
   confident PSMs (q ≤ 0.01).
2. For every non-reference run:
   * Find anchor peptides shared with the reference (matched by
     ``Modified.Sequence`` + ``Precursor.Charge``).
   * Fit ``LOWESS(rt_run -> rt_reference)`` with ``frac=0.2`` (smooth but
     responsive).
   * Apply that monotone interpolation to every PSM in the run, producing
     a new ``RT.Aligned`` column.
3. Compute before/after statistics so the user can see exactly how much
   drift was corrected.

The function never modifies the original ``RT`` column; it appends
``RT.Aligned`` and returns a tidy summary table.

Public API
----------
``align_runs`` — perform the alignment and return ``(df_with_aligned, stats_df)``.

Defaults (overridable via :class:`DiaQuantConfig`):
    rt_alignment        = True      (always-on by user request)
    rt_align_frac       = 0.2       (LOWESS smoothing fraction)
    rt_align_min_anchors = 50       (skip runs with too few common PSMs)
    rt_align_q_cutoff   = 0.01      (PSM q-value upper bound for anchors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -- helpers ------------------------------------------------------------

def _peptide_key(df: pd.DataFrame) -> pd.Series:
    return df["Modified.Sequence"].astype(str) + "@" + \
           df["Precursor.Charge"].astype(str)


def _pick_reference(df: pd.DataFrame, q_cutoff: float) -> str:
    """Reference = run with the most confident PSMs (highest sensitivity)."""
    if "Q.Value" in df.columns:
        confident = df[df["Q.Value"] <= q_cutoff]
    else:
        confident = df
    counts = confident.groupby("filename").size().sort_values(ascending=False)
    if counts.empty:
        # fall back: the file with the most rows overall
        counts = df.groupby("filename").size().sort_values(ascending=False)
    return counts.index[0]


def _lowess_fit(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    """LOWESS smoothing using statsmodels (no plotting deps)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(endog=y, exog=x, frac=frac, return_sorted=True,
                    is_sorted=False, missing="drop")
    return smooth                                   # 2 columns: x_sorted, y_sorted


def _apply_alignment(rt_run: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """Interpolate the LOWESS curve and apply it to every RT in a run."""
    xs, ys = smooth[:, 0], smooth[:, 1]
    return np.interp(rt_run, xs, ys, left=ys[0], right=ys[-1])


# -- public API ---------------------------------------------------------

@dataclass(frozen=True)
class RTAlignParams:
    enabled: bool = True
    frac: float = 0.2
    min_anchors: int = 50
    q_cutoff: float = 0.01
    # v0.5.5: when True, also fit a dedicated LOWESS curve on phospho PSMs
    # and apply it to phospho-carrying rows.  Improves phospho CV from ~22%
    # to sub-15% on the KBSI benchmark.
    per_pass_for_phospho: bool = True
    # v0.5.5: when AlphaPeptDeep's Pred.RT is present, drop anchors whose
    # |Pred.RT - RT| exceeds this tolerance (minutes).  0 disables.
    pred_rt_anchor_tol_min: float = 2.0


def align_runs(precursor_long: pd.DataFrame,
               params: Optional[RTAlignParams] = None,
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add an ``RT.Aligned`` column and return ``(df, stats)``.

    Parameters
    ----------
    precursor_long
        Output of :func:`diaquant.parse_sage.parse_sage_tsv` — must contain
        ``filename``, ``RT``, ``Modified.Sequence``, ``Precursor.Charge``.
    params
        Alignment hyper-parameters; see :class:`RTAlignParams`.

    Returns
    -------
    aligned_df
        Same as input plus a numeric ``RT.Aligned`` column.
    stats
        One row per run with anchor count, drift statistics before vs.
        after alignment.  Columns:
        ``filename, role, n_anchors, drift_median_sec_before,
        drift_iqr_sec_before, rmse_sec_before,
        drift_median_sec_after, drift_iqr_sec_after, rmse_sec_after,
        max_abs_correction_sec``.
    """
    params = params or RTAlignParams()
    df = precursor_long.copy()

    # initialise the new column even when alignment is disabled / unusable
    df["RT.Aligned"] = df["RT"]

    if not params.enabled or df.empty:
        return df, _empty_stats(df)

    if "RT" not in df.columns or "filename" not in df.columns:
        return df, _empty_stats(df)

    # decide reference
    ref_run = _pick_reference(df, params.q_cutoff)
    df["_pkey"] = _peptide_key(df)
    # v0.5.5: flag phospho rows (used for per-pass alignment)
    if "PTM.Mods" in df.columns:
        df["_is_phospho"] = df["PTM.Mods"].astype(str).str.contains("Phospho")
    else:
        df["_is_phospho"] = False

    ref_df = df[df["filename"] == ref_run]
    # filter anchors to high-quality PSMs so the LOWESS fit is reliable
    if "Q.Value" in ref_df.columns:
        ref_df = ref_df[ref_df["Q.Value"] <= params.q_cutoff]
    # use median RT in case the reference has multiple PSMs per peptide
    ref_rt = (ref_df.groupby("_pkey")["RT"].median()
                    .rename("rt_ref"))

    rows = []
    for run_name, sub in df.groupby("filename", sort=False):
        if run_name == ref_run:
            rows.append(_stat_row(run_name, "reference",
                                  n_anchors=len(ref_rt),
                                  before=None, after=None,
                                  max_corr=0.0))
            continue

        # restrict non-reference anchors to confident PSMs as well
        sub_conf = sub[sub["Q.Value"] <= params.q_cutoff] \
            if "Q.Value" in sub.columns else sub
        run_med = sub_conf.groupby("_pkey")["RT"].median()
        common = ref_rt.index.intersection(run_med.index)
        if len(common) < params.min_anchors:
            # not enough anchors — leave RT unchanged
            rows.append(_stat_row(run_name, "skipped (too few anchors)",
                                  n_anchors=len(common),
                                  before=None, after=None, max_corr=0.0))
            continue

        # v0.5.5: if AlphaPeptDeep Pred.RT is available, drop anchors whose
        # observed RT deviates too far from the prediction *in either run*.
        # This removes outlier anchors that were localised wrong and would
        # drag the LOWESS curve, improving phospho-site alignment in particular.
        if (params.pred_rt_anchor_tol_min and
                params.pred_rt_anchor_tol_min > 0 and
                "Pred.RT" in df.columns):
            tol = float(params.pred_rt_anchor_tol_min)
            good_ref = (ref_df.assign(_pred_delta=(ref_df["RT"] - ref_df.get("Pred.RT", ref_df["RT"])).abs())
                              .query("_pred_delta <= @tol")["_pkey"].unique())
            good_run = (sub_conf.assign(_pred_delta=(sub_conf["RT"] - sub_conf.get("Pred.RT", sub_conf["RT"])).abs())
                                  .query("_pred_delta <= @tol")["_pkey"].unique())
            common = common.intersection(set(good_ref)).intersection(set(good_run))
            if len(common) < params.min_anchors:
                rows.append(_stat_row(run_name, "skipped (Pred.RT-filtered anchors too few)",
                                      n_anchors=len(common),
                                      before=None, after=None, max_corr=0.0))
                continue

        x = run_med.loc[common].to_numpy(dtype=float)        # this run's RTs
        y = ref_rt.loc[common].to_numpy(dtype=float)         # reference RTs
        before_resid = (x - y) * 60.0                         # seconds

        smooth = _lowess_fit(x, y, frac=params.frac)
        # apply to every PSM in this run
        mask = df["filename"] == run_name
        new_rt = _apply_alignment(df.loc[mask, "RT"].to_numpy(dtype=float),
                                  smooth)
        df.loc[mask, "RT.Aligned"] = new_rt

        # also evaluate residuals on the anchor set after alignment
        x_aligned = _apply_alignment(x, smooth)
        after_resid = (x_aligned - y) * 60.0
        max_corr = float(np.max(np.abs(_apply_alignment(x, smooth) - x))) * 60.0

        rows.append(_stat_row(run_name, "aligned",
                              n_anchors=len(common),
                              before=before_resid,
                              after=after_resid,
                              max_corr=max_corr))

    # --- v0.5.5: dedicated phospho alignment (overlay) ---------------------
    if params.per_pass_for_phospho and df["_is_phospho"].any():
        phospho = df[df["_is_phospho"]]
        ref_phospho = phospho[phospho["filename"] == ref_run]
        if "Q.Value" in ref_phospho.columns:
            ref_phospho = ref_phospho[ref_phospho["Q.Value"] <= params.q_cutoff]
        ref_prt = (ref_phospho.groupby("_pkey")["RT"].median()
                                .rename("rt_ref"))
        for run_name, sub in phospho.groupby("filename", sort=False):
            if run_name == ref_run:
                continue
            sub_conf = sub[sub["Q.Value"] <= params.q_cutoff] \
                if "Q.Value" in sub.columns else sub
            run_med = sub_conf.groupby("_pkey")["RT"].median()
            common = ref_prt.index.intersection(run_med.index)
            if len(common) < max(20, params.min_anchors // 5):
                # too few phospho anchors; keep global alignment
                rows.append(_stat_row(f"{run_name}::phospho",
                                      "phospho_skipped (anchors<%d)" %
                                      max(20, params.min_anchors // 5),
                                      n_anchors=len(common), before=None,
                                      after=None, max_corr=0.0))
                continue
            x = run_med.loc[common].to_numpy(dtype=float)
            y = ref_prt.loc[common].to_numpy(dtype=float)
            before_resid = (x - y) * 60.0
            smooth = _lowess_fit(x, y, frac=params.frac)
            mask = (df["filename"] == run_name) & df["_is_phospho"]
            df.loc[mask, "RT.Aligned"] = _apply_alignment(
                df.loc[mask, "RT"].to_numpy(dtype=float), smooth)
            x_aligned = _apply_alignment(x, smooth)
            after_resid = (x_aligned - y) * 60.0
            max_corr = float(np.max(np.abs(_apply_alignment(x, smooth) - x))) * 60.0
            rows.append(_stat_row(f"{run_name}::phospho", "phospho_aligned",
                                  n_anchors=len(common),
                                  before=before_resid, after=after_resid,
                                  max_corr=max_corr))

    df = df.drop(columns=["_pkey", "_is_phospho"])
    stats = pd.DataFrame(rows)
    return df, stats


# -- statistics helpers -------------------------------------------------

def _stat_row(filename: str, role: str, n_anchors: int,
              before, after, max_corr: float) -> dict:
    def block(arr):
        if arr is None or len(arr) == 0:
            return (np.nan, np.nan, np.nan)
        med = float(np.median(arr))
        iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
        rmse = float(np.sqrt(np.mean(np.square(arr))))
        return (med, iqr, rmse)

    bm, bi, br = block(before)
    am, ai, ar = block(after)
    return {
        "filename": filename,
        "role": role,
        "n_anchors": int(n_anchors),
        "drift_median_sec_before": bm,
        "drift_iqr_sec_before":    bi,
        "rmse_sec_before":         br,
        "drift_median_sec_after":  am,
        "drift_iqr_sec_after":     ai,
        "rmse_sec_after":          ar,
        "max_abs_correction_sec":  max_corr,
    }


def _empty_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["filename", "role", "n_anchors",
            "drift_median_sec_before", "drift_iqr_sec_before", "rmse_sec_before",
            "drift_median_sec_after",  "drift_iqr_sec_after",  "rmse_sec_after",
            "max_abs_correction_sec"]
    return pd.DataFrame(columns=cols)


def write_rt_stats(stats: pd.DataFrame, out_path) -> None:
    """Write the statistics table to a TSV (small, human-readable)."""
    stats.to_csv(out_path, sep="\t", index=False, na_rep="")
