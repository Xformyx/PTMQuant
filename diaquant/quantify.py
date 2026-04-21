"""directLFQ-based quantification for precursors, protein groups and PTM sites.

We use the Apache-2.0 licensed `directlfq` package to perform the same
ratio-based normalization as MaxLFQ, in O(n) time.  directLFQ expects a long
table with ``protein``, ``ion`` (precursor), ``sample`` and ``intensity``
columns; here we provide three convenience wrappers that build that table for:

1. ``protein_quant``   – classical protein-group quantification
2. ``precursor_quant`` – simple per-precursor matrix (no LFQ; raw apex area)
3. ``site_quant``      – PTM-site-centric quantification, where each unique
                          (gene + site_position + mod_name) is treated as a
                          virtual "protein" so directLFQ rolls precursors up
                          to the modified site instead of to the parent
                          protein.  This is the central improvement over
                          DIA-NN's Top-1 site quantification.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import directlfq.lfq_manager as lfqm
import directlfq.utils as lfq_utils


def _to_lfq_input(df: pd.DataFrame, protein_col: str) -> pd.DataFrame:
    """Reshape a long PSM/precursor table into the wide format directLFQ wants.

    directLFQ v0.3 expects:
        index  : ['protein', 'ion']
        cols   : one column per sample (raw intensities, NaN for missing)
    """
    pivot = (df.pivot_table(index=[protein_col, "Precursor.Id"],
                            columns="filename",
                            values="Intensity",
                            aggfunc="max")
               .reset_index()
               .rename(columns={protein_col: "protein", "Precursor.Id": "ion"}))
    return pivot


def protein_quant(precursor_long: pd.DataFrame,
                  min_samples: int = 2) -> pd.DataFrame:
    """Roll precursor intensities up to protein-group LFQ values."""
    lfq_in = _to_lfq_input(precursor_long, protein_col="Protein.Group")
    out = lfqm.run_lfq(input_file=None, input_df=lfq_in,
                       min_nonan=min_samples,
                       num_samples_quadratic=10,
                       num_cores=0)
    return out


def site_quant(precursor_long: pd.DataFrame,
               min_samples: int = 2) -> pd.DataFrame:
    """Roll precursors up to PTM-site level.

    The trick is to build a synthetic ``protein`` identifier of the form
    ``GENE_S123_Phospho`` that uniquely encodes the (gene, residue, position,
    PTM type) tuple.  Several precursors that share the same site (different
    charge, missed cleavage, surrounding mods…) then collapse into the same
    LFQ value.
    """
    df = precursor_long.copy()
    df["__site__"] = (df["Genes"].fillna(df["Protein.Group"])
                      + "_" + df["PTM.Site.Positions"].fillna("none")
                      + "_" + df["PTM.Modification"].fillna("none"))
    # only modified precursors contribute to site quant
    df = df[df["PTM.Site.Positions"].fillna("none") != "none"]
    if df.empty:
        return pd.DataFrame()
    lfq_in = _to_lfq_input(df.rename(columns={"__site__": "Protein.Group"}),
                           protein_col="Protein.Group")
    return lfqm.run_lfq(input_file=None, input_df=lfq_in,
                        min_nonan=min_samples,
                        num_samples_quadratic=10,
                        num_cores=0)


def precursor_matrix(precursor_long: pd.DataFrame) -> pd.DataFrame:
    """Per-precursor wide matrix (raw apex area, no normalization)."""
    return (precursor_long.pivot_table(
        index=["Protein.Group", "Protein.Ids", "Protein.Names",
               "Genes", "First.Protein.Description", "Proteotypic",
               "Stripped.Sequence", "Modified.Sequence",
               "Precursor.Charge", "Precursor.Id"],
        columns="filename",
        values="Intensity",
        aggfunc="max")
        .reset_index())
