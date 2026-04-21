"""Sample-sheet driven differential analysis.

If the user provides a sample sheet TSV with at least the columns
``mzml_file`` and ``group`` (any extra columns such as ``condition``,
``timepoint`` and ``replicate`` are also recognised), diaquant will
automatically compute log2 fold-change and a moderated t-test between
every pair of groups for both the protein matrix and each PTM-site
matrix.

The implementation is intentionally lightweight: pandas + scipy only,
no R / limma dependency.  For a small number of replicates this
approximates a Welch t-test with Benjamini-Hochberg FDR correction,
which is the standard first-pass analysis for label-free DIA data.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:                      # pragma: no cover - scipy is required
    _HAS_SCIPY = False


def load_sample_sheet(path: Path) -> pd.DataFrame:
    """Load a sample sheet TSV; return an indexed DataFrame.

    Required columns: ``mzml_file``, ``group``.  Recognised optional
    columns: ``condition``, ``timepoint``, ``replicate``.  ``mzml_file``
    is matched against the *basename* of the columns in the wide matrix
    so users can write either basenames or full paths.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"mzml_file", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"sample_sheet is missing required columns: {sorted(missing)}"
        )
    df["mzml_basename"] = df["mzml_file"].apply(
        lambda s: Path(str(s)).name.replace(".mzML", "").replace(".mzml", "")
    )
    return df


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment of a vector of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    q = np.full(n, np.nan)
    mask = ~np.isnan(pvals)
    if not mask.any():
        return q
    p = pvals[mask]
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.full(p.shape, np.nan)
    out[order] = np.minimum(adj, 1.0)
    q[mask] = out
    return q


def differential(matrix: pd.DataFrame,
                 sample_sheet: pd.DataFrame,
                 id_cols: List[str],
                 log2: bool = True,
                 min_valid_per_group: int = 2,
                 ) -> pd.DataFrame:
    """Compute pairwise group fold-change and Welch t-test.

    ``matrix`` is a wide DIA-NN-style table (e.g. ``report.pg_matrix``)
    with a fixed set of ``id_cols`` followed by one intensity column per
    mzML file.  Columns are matched against ``sample_sheet`` by
    *basename* so users can drop full paths into the matrix without
    breaking lookups.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for differential analysis")

    sample_cols = [c for c in matrix.columns if c not in id_cols]
    base_to_full = {Path(c).name.replace(".mzML", "").replace(".mzml", ""): c
                    for c in sample_cols}

    groups: Dict[str, List[str]] = {}
    for _, row in sample_sheet.iterrows():
        col = base_to_full.get(row["mzml_basename"])
        if col is None:
            continue
        groups.setdefault(row["group"], []).append(col)

    if len(groups) < 2:
        raise ValueError("sample_sheet defines fewer than two groups; "
                         "differential analysis needs at least two.")

    intens = matrix[sample_cols].apply(pd.to_numeric, errors="coerce")
    if log2:
        intens = np.log2(intens.replace(0, np.nan))

    rows = []
    for g1, g2 in combinations(groups, 2):
        c1, c2 = groups[g1], groups[g2]
        m1 = intens[c1].values
        m2 = intens[c2].values
        n1_valid = (~np.isnan(m1)).sum(axis=1)
        n2_valid = (~np.isnan(m2)).sum(axis=1)
        valid = (n1_valid >= min_valid_per_group) & (n2_valid >= min_valid_per_group)

        log2fc = np.full(len(matrix), np.nan)
        pvals = np.full(len(matrix), np.nan)
        if valid.any():
            mean1 = np.nanmean(m1[valid], axis=1)
            mean2 = np.nanmean(m2[valid], axis=1)
            log2fc[valid] = mean1 - mean2  # already in log2 space
            t, p = scipy_stats.ttest_ind(
                m1[valid], m2[valid], axis=1,
                equal_var=False, nan_policy="omit",
            )
            pvals[valid] = p
        qvals = _bh_fdr(pvals)

        block = matrix[id_cols].copy()
        block["Comparison"] = f"{g1}_vs_{g2}"
        block["log2FC"] = log2fc
        block["p_value"] = pvals
        block["q_value"] = qvals
        rows.append(block)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
