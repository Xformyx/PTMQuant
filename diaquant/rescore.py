"""Post-hoc PSM rescoring with AlphaPeptDeep predictions (NEW in 0.5.0).

After each Sage pass finishes, :func:`rescore_with_predicted_library` joins
the parsed PSM table with the predicted spectral library produced by
:mod:`diaquant.predicted_library` and appends two new features:

* ``Pred.RT``        — AlphaPeptDeep-predicted retention time (minutes)
* ``Pred.RT.Delta``  — ``|RT - Pred.RT|`` in minutes
* ``Frag.Cosine``    — (optional) cosine similarity between the observed
                       MS2 spectrum and the AlphaPeptDeep-predicted
                       fragment intensity vector.  Only computed when
                       ``cfg.rescore_frag_cosine_cutoff > 0`` because it
                       requires re-parsing the mzML files.

The columns are preserved through RT alignment and ultimately written to
``report.tsv`` (see :mod:`diaquant.writer`).

Design notes
------------
* **Predicted RT replaces Sage's linear estimate.** Sage's built-in
  ``predicted_rt`` is a linear regression over the current run; AlphaPeptDeep
  is a pretrained deep model that generalises across gradients and PTMs.
  We therefore move Sage's original prediction into ``Predicted.RT.Sage``
  and overwrite ``Predicted.RT`` with the AlphaPeptDeep value so the
  existing ``rt_prediction_tolerance_min`` filter and LOWESS alignment
  automatically benefit from the better estimate.
* **RT tolerance is used for *demotion*, not deletion.** PSMs whose RT
  deviates from AlphaPeptDeep's prediction by more than
  ``cfg.rescore_rt_tol_min`` have their ``Score`` multiplied by 0.5, which
  lowers their rank in downstream de-duplication without removing them
  outright — the whole-proteome pass can still recover them.
* **Fragment cosine is gated and safe-by-default.** mzML re-parsing is
  expensive and may fail if the mzML path recorded in Sage's TSV is no
  longer valid; we therefore default ``rescore_frag_cosine_cutoff`` to 0.0
  (disabled) and any exception during fragment matching is logged but
  does not abort the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DiaQuantConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predicted-library parsing
# ---------------------------------------------------------------------------

def _load_predicted_library(library_tsv: Path) -> pd.DataFrame:
    """Read the AlphaPeptDeep-written TSV and return the minimal join frame.

    The TSV is the Spectronaut-compatible layout written by
    ``alphabase.spectral_library.translate.translate_to_tsv`` (one row per
    fragment).  For RT rescoring we only need ``ModifiedPeptide``,
    ``PrecursorCharge`` and ``iRT``/``RT`` per precursor, so we deduplicate.

    The actual column names differ slightly between AlphaBase versions; we
    therefore look for them case-insensitively and with a small set of
    aliases.  Returns an empty dataframe with the expected columns if the
    file is empty or unreadable.
    """
    columns = ["Modified.Sequence", "Precursor.Charge", "Pred.RT"]
    if not library_tsv.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(library_tsv, sep="\t", low_memory=False)
    except Exception as exc:                                         # pragma: no cover
        log.warning("Could not read predicted library %s (%s)", library_tsv, exc)
        return pd.DataFrame(columns=columns)

    # normalise column names
    lc = {c.lower(): c for c in df.columns}
    seq_col = (lc.get("modifiedpeptide") or lc.get("modified.sequence")
               or lc.get("modified_sequence"))
    charge_col = (lc.get("precursorcharge") or lc.get("precursor.charge")
                  or lc.get("precursor_charge") or lc.get("charge"))
    rt_col = (lc.get("rt") or lc.get("irt") or lc.get("normalizedretentiontime")
              or lc.get("rt_pred") or lc.get("predicted_rt"))
    if not (seq_col and charge_col and rt_col):                     # pragma: no cover
        log.warning(
            "Predicted library %s missing required columns "
            "(have: %s)", library_tsv, list(df.columns)[:8],
        )
        return pd.DataFrame(columns=columns)

    sub = (df[[seq_col, charge_col, rt_col]]
           .rename(columns={seq_col: "Modified.Sequence",
                            charge_col: "Precursor.Charge",
                            rt_col: "Pred.RT"})
           .drop_duplicates(["Modified.Sequence", "Precursor.Charge"]))
    # coerce types so pandas merge works
    sub["Precursor.Charge"] = pd.to_numeric(sub["Precursor.Charge"],
                                            errors="coerce").astype("Int64")
    sub["Pred.RT"] = pd.to_numeric(sub["Pred.RT"], errors="coerce")
    return sub.dropna(subset=["Pred.RT"])


# ---------------------------------------------------------------------------
# Modified.Sequence harmonisation
# ---------------------------------------------------------------------------
# Sage reports peptides as e.g. "AAAS[+79.9663]PEPR" (Δmass in brackets).
# AlphaPeptDeep / AlphaBase reports them as "AAAS[Phospho]PEPR" (UniMod
# modification name).  We therefore try to match on *mass-normalised*
# sequence: replace every modification string with the rounded Δmass.

import re

_SAGE_MASS_RE = re.compile(r"\[([+-]?\d+\.\d+)\]")
_NAME_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")

# crude UniMod name -> Δmass lookup for the built-in PTMs; used when Sage
# has a numeric bracket but the library has a name and vice versa.
_NAME_TO_MASS = {
    "Carbamidomethyl": 57.021464,
    "Oxidation": 15.994915,
    "Acetyl": 42.010565,
    "Acetyl_Nterm": 42.010565,
    "Phospho": 79.966331,
    "GlyGly": 114.042927,
    "GG": 114.042927,
    "Methyl": 14.015650,
    "Dimethyl": 28.031300,
    "Trimethyl": 42.046950,
    "Succinyl": 100.016044,
    "Malonyl": 86.000394,
    "Crotonyl": 68.026215,
    "Sumo_QQTGG": 471.207606,
    "QQTGG": 471.207606,
    "Citrullination": 0.984016,
    "Deamidated": 0.984016,
}


def _canonical_mod_key(mod_seq: str) -> str:
    """Return a normalised peptide sequence usable as a join key.

    Both Δmass-style (``[+79.9663]``) and name-style (``[Phospho]``) are
    collapsed to ``[+79.97]`` (2-decimal rounded mass) so the Sage PSM
    table and the AlphaPeptDeep library can be joined even though each
    speaks a slightly different dialect.
    """
    if not isinstance(mod_seq, str):
        return ""

    def repl_mass(match: re.Match) -> str:
        try:
            return f"[{float(match.group(1)):+.2f}]"
        except ValueError:                                          # pragma: no cover
            return match.group(0)

    def repl_name(match: re.Match) -> str:
        name = match.group(1)
        mass = _NAME_TO_MASS.get(name)
        if mass is None:
            # unknown name: keep as-is so different names don't collide
            return f"[{name}]"
        return f"[{mass:+.2f}]"

    out = _SAGE_MASS_RE.sub(repl_mass, mod_seq)
    out = _NAME_RE.sub(repl_name, out)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rescore_with_predicted_library(
    psm_df: pd.DataFrame,
    library_tsv: Optional[Path],
    cfg: DiaQuantConfig,
) -> pd.DataFrame:
    """Join predictions, add features, optionally demote outliers.

    Parameters
    ----------
    psm_df
        Long PSM table produced by :func:`diaquant.parse_sage.parse_sage_tsv`.
        Must at least contain ``Modified.Sequence``, ``Precursor.Charge``,
        ``RT`` (minutes).  Optional columns ``Predicted.RT`` and ``Score``
        are used when present.
    library_tsv
        Path to the predicted library TSV; ``None`` (or a missing file)
        short-circuits the function — the PSM frame is returned unchanged.
    cfg
        Active diaquant config.  ``rescore_with_prediction``,
        ``rescore_rt_tol_min`` and ``rescore_frag_cosine_cutoff`` are the
        relevant keys.

    Returns
    -------
    pandas.DataFrame
        The same frame with new columns ``Pred.RT``, ``Pred.RT.Delta``
        and (when fragment cosine is enabled) ``Frag.Cosine``.
    """
    if psm_df is None or psm_df.empty:
        return psm_df
    if not cfg.rescore_with_prediction:
        return psm_df
    if library_tsv is None:
        log.info("[diaquant] rescore skipped: no predicted library available.")
        return psm_df

    lib_df = _load_predicted_library(Path(library_tsv))
    if lib_df.empty:
        log.info("[diaquant] rescore skipped: predicted library is empty.")
        return psm_df

    psm = psm_df.copy()
    psm["_join_seq"] = psm["Modified.Sequence"].map(_canonical_mod_key)
    lib = lib_df.copy()
    lib["_join_seq"] = lib["Modified.Sequence"].map(_canonical_mod_key)
    lib = lib.drop(columns=["Modified.Sequence"])

    # cast the charge on both sides to plain int for a clean merge
    psm["Precursor.Charge"] = pd.to_numeric(psm["Precursor.Charge"],
                                            errors="coerce").astype("Int64")

    merged = psm.merge(
        lib[["_join_seq", "Precursor.Charge", "Pred.RT"]],
        on=["_join_seq", "Precursor.Charge"],
        how="left",
    )
    n_hit = merged["Pred.RT"].notna().sum()
    log.info(
        "[diaquant] rescore: matched %d / %d PSMs to predicted RT (%.1f%%)",
        n_hit, len(merged), 100.0 * n_hit / max(len(merged), 1),
    )

    # Preserve Sage's own prediction for provenance then overwrite with
    # AlphaPeptDeep's value when available (falls back to Sage's when not).
    if "Predicted.RT" in merged.columns:
        merged["Predicted.RT.Sage"] = merged["Predicted.RT"]
        merged["Predicted.RT"] = merged["Pred.RT"].fillna(merged["Predicted.RT"])
    else:
        merged["Predicted.RT"] = merged["Pred.RT"]

    merged["Pred.RT.Delta"] = (merged["RT"] - merged["Pred.RT"]).abs()

    # Demote (not delete) PSMs whose RT is far from the prediction.  A
    # multiplicative factor on ``Score`` keeps deduplication well-behaved
    # without dropping rows that the whole-proteome pass might still want.
    if cfg.rescore_rt_tol_min is not None and "Score" in merged.columns:
        tol = float(cfg.rescore_rt_tol_min)
        outliers = merged["Pred.RT.Delta"] > tol
        if outliers.any():
            log.info(
                "[diaquant] rescore: demoting %d PSMs with |Pred.RT.Delta| > %.1f min",
                int(outliers.sum()), tol,
            )
            merged.loc[outliers, "Score"] = merged.loc[outliers, "Score"] * 0.5

    merged = merged.drop(columns=["_join_seq"])

    # ----- optional fragment cosine (disabled by default) --------------
    cutoff = float(cfg.rescore_frag_cosine_cutoff or 0.0)
    if cutoff > 0:
        merged = _augment_fragment_cosine(merged, library_tsv, cfg, cutoff)

    return merged


# ---------------------------------------------------------------------------
# Optional fragment cosine (requires pyteomics + mzML re-parsing)
# ---------------------------------------------------------------------------

def _augment_fragment_cosine(
    psm: pd.DataFrame,
    library_tsv: Path,
    cfg: DiaQuantConfig,
    cutoff: float,
) -> pd.DataFrame:
    """Attach ``Frag.Cosine`` by re-reading observed MS2 spectra.

    This is an opt-in feature (``cfg.rescore_frag_cosine_cutoff > 0``) because
    it re-parses every mzML file referenced by the PSM table.  If pyteomics
    or the mzML files are unavailable, the function logs a warning and
    returns the PSM table unchanged.
    """
    try:                                                             # pragma: no cover
        from pyteomics import mzml
    except Exception as exc:                                         # pragma: no cover
        log.warning("Fragment cosine disabled: pyteomics unavailable (%s)", exc)
        return psm

    try:                                                             # pragma: no cover
        lib = pd.read_csv(library_tsv, sep="\t", low_memory=False)
    except Exception as exc:                                         # pragma: no cover
        log.warning("Fragment cosine disabled: cannot read library (%s)", exc)
        return psm

    # normalise library column names so we can look up fragments per precursor
    lc = {c.lower(): c for c in lib.columns}
    seq_col = lc.get("modifiedpeptide") or lc.get("modified.sequence")
    charge_col = lc.get("precursorcharge") or lc.get("precursor.charge")
    fmz_col = lc.get("productmz") or lc.get("fragment_mz") or lc.get("fragmentmz")
    fint_col = lc.get("libraryintensity") or lc.get("fragment_intensity") or \
               lc.get("relativefragmentintensity")
    if not all([seq_col, charge_col, fmz_col, fint_col]):            # pragma: no cover
        log.warning("Fragment cosine disabled: library missing fragment columns.")
        return psm

    # build a (canonical_seq, charge) -> DataFrame lookup for fast access
    lib["_k"] = lib[seq_col].map(_canonical_mod_key)
    lib_group = dict(list(lib.groupby(["_k", charge_col])))
    log.info("[diaquant] rescore: computing fragment cosine for up to %d PSMs",
             len(psm))

    cosines = np.full(len(psm), np.nan, dtype=float)
    psm = psm.reset_index(drop=True)

    # group PSMs by mzML file so we only open each once
    for fname, sub in psm.groupby("filename"):
        mzml_path = Path(str(fname))
        if not mzml_path.exists():                                   # pragma: no cover
            log.warning("Fragment cosine: mzML missing (%s)", mzml_path)
            continue
        try:                                                         # pragma: no cover
            reader = mzml.read(str(mzml_path))
            scans = {int(s.get("id", "").split("scan=")[-1]): s for s in reader}
        except Exception as exc:                                     # pragma: no cover
            log.warning("Fragment cosine: cannot parse %s (%s)", mzml_path, exc)
            continue

        for idx, row in sub.iterrows():                              # pragma: no cover
            scan_no = int(row.get("Scan.Number", -1) or -1)
            spec = scans.get(scan_no)
            if spec is None:
                continue
            key = (_canonical_mod_key(row["Modified.Sequence"]),
                   int(row["Precursor.Charge"]))
            lib_rows = lib_group.get(key)
            if lib_rows is None or lib_rows.empty:
                continue
            obs_mz = np.asarray(spec.get("m/z array", []), dtype=float)
            obs_int = np.asarray(spec.get("intensity array", []), dtype=float)
            if obs_mz.size == 0:
                continue
            pred_mz = lib_rows[fmz_col].to_numpy(dtype=float)
            pred_int = lib_rows[fint_col].to_numpy(dtype=float)
            cosines[idx] = _cosine(obs_mz, obs_int, pred_mz, pred_int,
                                   ppm=cfg.fragment_tol_ppm)

    psm["Frag.Cosine"] = cosines
    dropped = (psm["Frag.Cosine"] < cutoff) & psm["Frag.Cosine"].notna()
    if dropped.any():                                                # pragma: no cover
        log.info("[diaquant] rescore: dropping %d PSMs with Frag.Cosine < %.2f",
                 int(dropped.sum()), cutoff)
        psm = psm[~dropped].reset_index(drop=True)
    return psm


def _cosine(obs_mz: np.ndarray, obs_int: np.ndarray,
            pred_mz: np.ndarray, pred_int: np.ndarray,
            ppm: float = 12.0) -> float:                             # pragma: no cover
    """Cosine similarity of observed vs predicted fragment intensities.

    Peaks in ``pred_mz`` are matched to observed peaks within ``ppm``
    tolerance.  Unmatched predicted peaks contribute zero observed
    intensity and vice versa.  Uses L2-normalised dot product.
    """
    if obs_mz.size == 0 or pred_mz.size == 0:
        return float("nan")
    matched_obs = np.zeros_like(pred_int)
    tol = ppm * 1e-6
    # naive O(P*O) match; predictions are <= 50 per peptide so this is fine
    for i, pm in enumerate(pred_mz):
        window = tol * pm
        mask = np.abs(obs_mz - pm) <= window
        if mask.any():
            matched_obs[i] = obs_int[mask].max()
    if matched_obs.sum() == 0:
        return 0.0
    v1 = matched_obs / np.linalg.norm(matched_obs)
    v2 = pred_int / (np.linalg.norm(pred_int) or 1.0)
    return float(np.dot(v1, v2))
