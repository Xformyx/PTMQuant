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

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_lfq_input(df: pd.DataFrame, protein_col: str) -> pd.DataFrame:
    """Reshape a long PSM/precursor table into the wide format directLFQ wants.

    directLFQ expects:
        columns: ['protein', 'ion', <sample1>, <sample2>, ...]
        intensities: raw (not log-transformed) – directLFQ log-transforms internally
    """
    pivot = (df.pivot_table(index=[protein_col, "Precursor.Id"],
                            columns="filename",
                            values="Intensity",
                            aggfunc="max")
               .reset_index()
               .rename(columns={protein_col: "protein", "Precursor.Id": "ion"}))
    return pivot


def _run_lfq_on_df(lfq_in: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Run directLFQ normalization + protein estimation directly on a DataFrame.

    Uses the low-level directLFQ API to avoid writing temporary files to disk.
    Returns a DataFrame with protein-level LFQ intensities (one column per sample).
    """
    import directlfq.utils as lfqutils
    import directlfq.normalization as lfqnorm
    import directlfq.protein_intensity_estimation as lfqprot
    import directlfq.config as lfqcfg

    lfqcfg.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
    lfqcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas()

    df = lfqutils.sort_input_df_by_protein_and_quant_id(lfq_in)
    df = lfqutils.remove_potential_quant_id_duplicates(df)
    df = lfqutils.index_and_log_transform_input_df(df)
    df = lfqutils.remove_allnan_rows_input_df(df)

    if df.empty:
        return pd.DataFrame()

    try:
        df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
            df, num_samples_quadratic=50
        ).complete_dataframe
    except Exception as exc:
        logger.warning("directLFQ normalization failed (%s); skipping normalization.", exc)

    protein_df, _ = lfqprot.estimate_protein_intensities(
        df,
        min_nonan=min_samples,
        num_samples_quadratic=10,
        num_cores=None,
    )
    return protein_df


def protein_quant(precursor_long: pd.DataFrame,
                  min_samples: int = 1) -> pd.DataFrame:
    """Roll precursor intensities up to protein-group LFQ values."""
    lfq_in = _to_lfq_input(precursor_long, protein_col="Protein.Group")
    return _run_lfq_on_df(lfq_in, min_samples=min_samples)


def site_quant(precursor_long: pd.DataFrame,
               min_samples: int = 1) -> pd.DataFrame:
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
    # Drop original Protein.Group before renaming __site__ to avoid duplicate columns
    site_df = (df.drop(columns=["Protein.Group"], errors="ignore")
                 .rename(columns={"__site__": "Protein.Group"}))
    lfq_in = _to_lfq_input(site_df, protein_col="Protein.Group")
    return _run_lfq_on_df(lfq_in, min_samples=min_samples)


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
