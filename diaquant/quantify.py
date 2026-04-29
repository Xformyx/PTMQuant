"""directLFQ-based quantification for precursors, protein groups and PTM sites.

We use the Apache-2.0 licensed `directlfq` package to perform the same
ratio-based normalization as MaxLFQ, in O(n) time.  directLFQ expects a long
table with ``protein``, ``ion`` (precursor), ``sample`` and ``intensity``
columns; here we provide three convenience wrappers that build that table for:

1. ``protein_quant``   \u2013 classical protein-group quantification
2. ``precursor_quant`` \u2013 simple per-precursor matrix (no LFQ; raw apex area)
3. ``site_quant``      \u2013 PTM-site-centric quantification, where each unique
                          (accession, residue, absolute_pos, mod_name) tuple
                          is treated as a virtual "protein" so directLFQ rolls
                          precursors up to the modified site instead of to the
                          parent protein.  This is the central improvement over
                          DIA-NN's Top-1 site quantification.

v0.5.3 changes
--------------
* ``site_quant`` now computes the *absolute* residue position in the parent
  protein instead of the peptide-local position that 0.5.1 emitted.  This
  required a FASTA lookup which the caller threads in via
  ``df.attrs['fasta_records']`` (populated by ``attach_fasta_meta``).
* The site key now encodes the Protein.Group accession so two different genes
  that happen to share the same modified residue do not collapse into one row.
* ``localization_cutoff`` is honoured here as well: precursors whose
  ``Best.Site.Probability`` is below the cutoff are excluded from the site
  roll-up unless ``include_low_loc`` is set.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_lfq_input(df: pd.DataFrame, protein_col: str) -> pd.DataFrame:
    """Reshape a long PSM/precursor table into the wide format directLFQ wants."""
    pivot = (df.pivot_table(index=[protein_col, "Precursor.Id"],
                            columns="filename",
                            values="Intensity",
                            aggfunc="max")
               .reset_index()
               .rename(columns={protein_col: "protein", "Precursor.Id": "ion"}))
    return pivot


def _run_lfq_on_df(lfq_in: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Run directLFQ normalization + protein estimation directly on a DataFrame."""
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


# ---------------------------------------------------------------------------
# v0.5.3 site-matrix bugfix: compute absolute residue position per precursor
# ---------------------------------------------------------------------------

# ``PTM.Site.Positions`` produced by ptm_localization is a ``;``-separated
# string of ``<AA><peptide_local_pos>`` tokens (e.g. ``"S3;T7"``).  The 0.5.1
# site matrix treated these tokens as absolute protein positions, which is
# wrong for every peptide that does not start at protein position 1 and was
# also missing the protein accession, so two different genes that happened to
# share the same (residue, peptide_local_pos, mod) collapsed into one row.
_SITE_TOKEN_RE = re.compile(r"^([A-Z])(\d+)$")


def _abs_site_keys(stripped_seq: str,
                   site_positions: str,
                   accession: str,
                   gene: str,
                   mod_name: str,
                   fasta_records: Optional[dict]) -> list[str]:
    """Return the list of site keys for one precursor.

    Each key encodes ``<Protein.Group>|<AA><abs_pos>|<Mod>``.  When the parent
    protein sequence is unknown (no FASTA records or accession missing) we
    fall back to the peptide-local position and append a ``_local`` suffix so
    the ambiguity is obvious to downstream consumers.
    """
    if not site_positions or site_positions == "none":
        return []
    keys: list[str] = []
    seq = ""
    if fasta_records and accession in fasta_records:
        seq = fasta_records[accession].get("seq", "") or ""
    # locate peptide in protein sequence (1-based position of its first residue)
    pep_start = seq.find(stripped_seq) + 1 if seq and stripped_seq else 0
    label = gene or accession or "unknown"
    for token in str(site_positions).split(";"):
        m = _SITE_TOKEN_RE.match(token.strip())
        if not m:
            continue
        aa, pep_pos = m.group(1), int(m.group(2))
        if pep_start > 0:
            abs_pos = pep_start + pep_pos - 1
            site = f"{aa}{abs_pos}"
        else:
            # fallback: peptide-local position; mark with a suffix so a
            # downstream report can flag it for manual review.
            site = f"{aa}{pep_pos}_local"
        keys.append(f"{accession}|{label}|{site}|{mod_name}")
    return keys


def site_quant(precursor_long: pd.DataFrame,
               min_samples: int = 1,
               localization_cutoff: float = 0.0,
               include_low_loc: bool = False) -> pd.DataFrame:
    """Roll precursors up to PTM-site level using absolute protein positions.

    Parameters
    ----------
    precursor_long
        Long precursor table with ``Modified.Sequence``, ``Stripped.Sequence``,
        ``PTM.Site.Positions``, ``PTM.Modification``, ``Protein.Group``,
        ``Genes``, ``Intensity``, ``filename``, ``Best.Site.Probability``.
    min_samples
        Minimum non-NaN samples per site (directLFQ parameter).
    localization_cutoff
        When > 0 and ``include_low_loc`` is False, precursors whose
        ``Best.Site.Probability`` is below this cutoff are excluded from the
        site roll-up (the new v0.5.3 phospho localization filter).
    include_low_loc
        Set to True to keep low-localization PSMs despite the cutoff; this
        mirrors DIA-NN's permissive behaviour and is useful for downstream
        tools that do their own site-probability filtering.
    """
    df = precursor_long.copy()
    if df.empty:
        return pd.DataFrame()

    # Only modified precursors contribute to site quant.
    df = df[df["PTM.Site.Positions"].fillna("none") != "none"]
    if df.empty:
        return pd.DataFrame()

    # v0.5.3 localization-probability filter.  When the upstream PSM table
    # does not carry Best.Site.Probability we keep every row (fail-open).
    if (localization_cutoff and localization_cutoff > 0.0
            and not include_low_loc
            and "Best.Site.Probability" in df.columns):
        before = len(df)
        df = df[df["Best.Site.Probability"].fillna(0.0) >= localization_cutoff]
        logger.info(
            "site_quant: localization filter (\u2265%.2f): %d \u2192 %d precursors",
            localization_cutoff, before, len(df),
        )
    if df.empty:
        return pd.DataFrame()

    # FASTA records attached by attach_fasta_meta() so we can compute the
    # absolute residue position without re-reading the FASTA here.
    fasta_records = precursor_long.attrs.get("fasta_records") \
        if hasattr(precursor_long, "attrs") else None

    # v0.5.3.1: Protein.Group is the full ``sp|P12345|GENE_HUMAN`` Sage tag,
    # but fasta_records is keyed on the bare UniProt accession.  Translate
    # each Protein.Group to its accession before the FASTA lookup; otherwise
    # _abs_site_keys() falls back to ``_local`` and the writer drops the row.
    extract_acc = (
        precursor_long.attrs.get("protein_group_accession")
        if hasattr(precursor_long, "attrs") else None
    )
    if extract_acc is None:
        # Fall back to in-module helper if attach_fasta_meta() never ran.
        from .parse_sage import _extract_accession as extract_acc

    df["__site_keys__"] = df.apply(
        lambda r: _abs_site_keys(
            stripped_seq=str(r.get("Stripped.Sequence", "")),
            site_positions=str(r.get("PTM.Site.Positions", "")),
            accession=extract_acc(str(r.get("Protein.Group", ""))),
            gene=str(r.get("Genes", "") or ""),
            mod_name=str(r.get("PTM.Modification", "none")),
            fasta_records=fasta_records,
        ),
        axis=1,
    )
    # Explode so one precursor contributing to two distinct sites (rare, but
    # happens for multi-modified peptides) lands in both rows.
    df = df.explode("__site_keys__")
    df = df[df["__site_keys__"].notna() & (df["__site_keys__"] != "")]
    if df.empty:
        return pd.DataFrame()

    site_df = (df.drop(columns=["Protein.Group"], errors="ignore")
                 .rename(columns={"__site_keys__": "Protein.Group"}))
    lfq_in = _to_lfq_input(site_df, protein_col="Protein.Group")
    site_lfq = _run_lfq_on_df(lfq_in, min_samples=min_samples)

    # Attach the mean localization probability per site for the writer.
    if "Best.Site.Probability" in df.columns and not site_lfq.empty:
        loc = (df.groupby(df["__site_keys__"].rename("protein")
                          if False else "Protein.Group")  # keep pandas happy
                 ["Best.Site.Probability"].mean()
                 .rename("Best.Site.Probability"))
        loc.index.name = "protein"
        site_lfq = site_lfq.merge(loc, left_on="protein",
                                  right_index=True, how="left")
    return site_lfq


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
