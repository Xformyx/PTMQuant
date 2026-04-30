"""Group-aware missing-value imputation for the precursor / pg / site matrices.

NEW in v0.5.8.

Why we cannot just use a global KNN imputer
-------------------------------------------
DIA proteomics matrices have a strong "missing-not-at-random" structure: the
missingness is dominated by ion intensity falling below the MS2 detection
limit in some samples, which itself correlates with the experimental group
(e.g. control vs treatment).  Imputing across groups therefore destroys the
biological signal we actually want to measure.

Strategy
~~~~~~~~
For each row (precursor, protein group or PTM site):

* Group sample columns by the ``group`` column of the user's sample sheet.
* Within each group, count the number of non-missing observations.  If a row
  has at least :attr:`min_obs_per_group` valid values in a group, fill the
  remaining samples in that group with a per-group statistic
  (median by default).  The cells we fill are flagged as imputed.
* If a group has zero valid values for that row, we leave the missing cells
  untouched - we never invent biology where none exists.

A new column ``Intensity.Imputed.Frac`` is added so downstream tools (Perseus,
limma, PTM-platform R notebook) can re-filter rows whose proportion of
imputed values exceeds a chosen threshold.

The default behaviour is opt-in (``cfg.impute_method == "none"``) so existing
v0.5.7 pipelines see no change unless the user explicitly enables it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImputeParams:
    """Parameters controlling group-aware imputation."""

    #: ``"none"`` (default), ``"group_median"``, ``"group_min"`` or ``"knn"``.
    method: str = "none"
    #: Minimum number of valid observations per group required to fill the
    #: remaining missing cells in that group.
    min_obs_per_group: int = 2
    #: KNN-only: number of nearest rows used as donors.
    knn_n_neighbors: int = 5
    #: KNN-only: weighting scheme passed to ``sklearn.impute.KNNImputer``.
    knn_weights: str = "distance"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def impute_matrix(matrix: pd.DataFrame,
                  sample_to_group: dict[str, str],
                  id_cols: List[str],
                  params: Optional[ImputeParams] = None,
                  ) -> pd.DataFrame:
    """Return a copy of ``matrix`` with missing sample cells imputed.

    Parameters
    ----------
    matrix
        DataFrame with identity columns (``id_cols``) followed by sample
        columns of intensities.  Missing values may be ``NaN`` or 0.
    sample_to_group
        Maps each sample column name in ``matrix`` to its experimental group
        label (typically ``stats.load_sample_sheet(...)`` -> ``mzml_basename``
        -> ``group``).  Sample columns absent from the map are treated as a
        single "ungrouped" pseudo-group.
    id_cols
        Identity columns to preserve (e.g. ``["Protein.Group", "Genes",
        "PTM.Site", "PTM.Modification", "Best.Site.Probability"]``).
    params
        :class:`ImputeParams`.  ``method="none"`` returns ``matrix.copy()``
        unchanged (still adds the all-zero ``Intensity.Imputed.Frac`` column).

    Returns
    -------
    pandas.DataFrame
        Same column order as ``matrix`` plus a new
        ``Intensity.Imputed.Frac`` column inserted after the identity block.
    """
    params = params or ImputeParams()
    out = matrix.copy()
    sample_cols = [c for c in out.columns if c not in id_cols]
    if not sample_cols:
        out["Intensity.Imputed.Frac"] = 0.0
        return out

    # Treat 0 and empty strings as NaN so the imputer counts them properly.
    out[sample_cols] = out[sample_cols].replace({0: np.nan, "": np.nan})
    out[sample_cols] = out[sample_cols].apply(pd.to_numeric, errors="coerce")

    if params.method == "none":
        out["Intensity.Imputed.Frac"] = 0.0
        return _reorder(out, id_cols, sample_cols)

    # Mask of cells that are NaN BEFORE imputation - used to compute the
    # imputed-fraction column and to set the "imputed" flag.
    nan_mask_before = out[sample_cols].isna()

    if params.method in ("group_median", "group_min"):
        out = _impute_group_stat(
            out, sample_cols, sample_to_group, params,
        )
    elif params.method == "knn":
        out = _impute_knn(out, sample_cols, sample_to_group, params)
    else:
        raise ValueError(
            f"Unknown impute_method: {params.method!r}. "
            f"Choose 'none', 'group_median', 'group_min' or 'knn'."
        )

    nan_mask_after = out[sample_cols].isna()
    n_imputed = (nan_mask_before & ~nan_mask_after).sum(axis=1)
    out["Intensity.Imputed.Frac"] = (
        n_imputed / max(1, len(sample_cols))
    ).astype(float)

    n_total = int((nan_mask_before & ~nan_mask_after).sum().sum())
    n_left  = int(nan_mask_after.sum().sum())
    logger.info(
        "imputation: method=%s filled %d cells (%d still missing)",
        params.method, n_total, n_left,
    )

    return _reorder(out, id_cols, sample_cols)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _reorder(df: pd.DataFrame, id_cols: List[str],
             sample_cols: List[str]) -> pd.DataFrame:
    return df[[*id_cols, "Intensity.Imputed.Frac", *sample_cols]]


def _group_columns(sample_cols: List[str],
                   sample_to_group: dict[str, str]
                   ) -> dict[str, List[str]]:
    """Return ``{group_label: [sample_col, ...]}`` (deterministic order)."""
    groups: dict[str, List[str]] = {}
    for col in sample_cols:
        g = sample_to_group.get(col, "__ungrouped__")
        groups.setdefault(g, []).append(col)
    return groups


def _impute_group_stat(out: pd.DataFrame,
                       sample_cols: List[str],
                       sample_to_group: dict[str, str],
                       params: ImputeParams) -> pd.DataFrame:
    groups = _group_columns(sample_cols, sample_to_group)
    for label, cols in groups.items():
        if len(cols) < 2:
            # No imputation possible from a single sample.
            continue
        block = out[cols]
        n_valid = block.notna().sum(axis=1)
        eligible = n_valid >= params.min_obs_per_group
        if not eligible.any():
            continue
        if params.method == "group_median":
            stat = block.median(axis=1, skipna=True)
        else:  # "group_min"
            stat = block.min(axis=1, skipna=True)
        # Fill only the missing cells in eligible rows of this group.
        for c in cols:
            mask = eligible & out[c].isna()
            if mask.any():
                out.loc[mask, c] = stat.loc[mask]
    return out


def _impute_knn(out: pd.DataFrame,
                sample_cols: List[str],
                sample_to_group: dict[str, str],
                params: ImputeParams) -> pd.DataFrame:
    """Per-group KNN imputation using ``sklearn.impute.KNNImputer``."""
    try:                                                       # pragma: no cover
        from sklearn.impute import KNNImputer
    except ImportError as exc:                                 # pragma: no cover
        logger.warning(
            "scikit-learn not installed (%s); KNN imputation unavailable. "
            "Install with `pip install scikit-learn` or pick "
            "'group_median' / 'group_min' instead.", exc,
        )
        return out

    groups = _group_columns(sample_cols, sample_to_group)
    for label, cols in groups.items():
        if len(cols) < 2:
            continue
        block = out[cols].to_numpy(dtype=float)
        # Skip the imputer for fully-missing groups - it would emit NaN-filled
        # arrays back and waste cycles.
        any_value = np.isfinite(block).any(axis=1)
        if not any_value.any():
            continue
        # Restrict KNN to rows that meet min_obs_per_group; everything else
        # stays missing (we do not borrow strength from sparse rows).
        n_valid = np.isfinite(block).sum(axis=1)
        eligible = n_valid >= params.min_obs_per_group
        if not eligible.any():
            continue
        sub = block[eligible]
        imputer = KNNImputer(
            n_neighbors=int(params.knn_n_neighbors),
            weights=params.knn_weights,
        )
        try:
            filled = imputer.fit_transform(sub)
        except Exception as exc:                               # pragma: no cover
            logger.warning(
                "KNN imputation failed in group %s (%s); leaving NaNs.",
                label, exc,
            )
            continue
        sub_idx = np.where(eligible)[0]
        for j, c in enumerate(cols):
            new_col = block[:, j].copy()
            new_col[sub_idx] = filled[:, j]
            out[c] = new_col
    return out


def build_sample_to_group(sample_cols: List[str],
                          sample_sheet: Optional[pd.DataFrame]
                          ) -> dict[str, str]:
    """Map matrix sample-column names -> ``group`` label.

    ``sample_sheet`` is the DataFrame produced by
    :func:`diaquant.stats.load_sample_sheet` and is expected to carry at least
    ``mzml_basename`` and ``group`` columns.  Sample columns that cannot be
    matched are returned with the special label ``"__ungrouped__"`` so the
    imputer collapses them into a single residual group instead of crashing.
    """
    if sample_sheet is None or sample_sheet.empty:
        return {c: "__ungrouped__" for c in sample_cols}
    if "mzml_basename" not in sample_sheet.columns or \
       "group" not in sample_sheet.columns:
        return {c: "__ungrouped__" for c in sample_cols}
    mapping = dict(zip(sample_sheet["mzml_basename"].astype(str),
                       sample_sheet["group"].astype(str)))
    out = {}
    for col in sample_cols:
        # Match by exact basename or by any case-insensitive sheet entry that
        # is a substring of the column name (supports DIA-NN style absolute
        # paths inside the matrix header).
        if col in mapping:
            out[col] = mapping[col]
            continue
        match = next((g for k, g in mapping.items()
                      if k and k.lower() in col.lower()), None)
        out[col] = match or "__ungrouped__"
    return out
