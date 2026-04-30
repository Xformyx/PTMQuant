"""directLFQ-based quantification for precursors, protein groups and PTM sites.

We use the Apache-2.0 licensed `directlfq` package to perform the same
ratio-based normalization as MaxLFQ, in O(n) time.  directLFQ expects a long
table with ``protein``, ``ion`` (precursor), ``sample`` and ``intensity``
columns; here we provide three convenience wrappers that build that table for:

1. ``protein_quant``   – classical protein-group quantification
2. ``precursor_quant`` – simple per-precursor matrix (no LFQ; raw apex area)
3. ``site_quant``      – PTM-site-centric quantification, where each unique
                         ``(accession, residue, abs_pos, mod_name)`` tuple
                         is treated as a virtual "protein" so directLFQ rolls
                         precursors up to the modified site instead of to the
                         parent protein.  This is the central improvement over
                         DIA-NN's Top-1 site quantification.

v0.5.5 changes
--------------
* The site roll-up no longer relies on the positionally ambiguous
  ``PTM.Site.Positions`` + ``PTM.Modification`` pair (which collapsed all
  modifications on a peptide into one mod name and was the root cause of
  the 0-rows ``ptm_site_matrix.tsv`` regression).  Instead it consumes the
  richer ``PTM.Mods`` column produced by the rewritten
  :mod:`diaquant.ptm_localization` module, which carries one
  ``(mod_name, residue, peptide_local_pos, probability)`` triple per site.
* ``site_quant`` now exposes a ``phospho_only`` parameter (default
  ``True``) that keeps only phospho sites in the output.  Set to
  ``False`` to emit every modification.
* The per-site localization probability written to the matrix is the
  actual probability of *that site* (not the peptide-wide max), which
  lets downstream tools such as PTMsite-Finder filter individual rows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ptm_localization import _normalise_tag, iter_site_entries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# directLFQ helpers
# ---------------------------------------------------------------------------

def _to_lfq_input(df: pd.DataFrame, protein_col: str,
                  ion_col: str = "Precursor.Id") -> pd.DataFrame:
    """Reshape a long PSM/precursor table into the wide format directLFQ wants."""
    pivot = (df.pivot_table(index=[protein_col, ion_col],
                            columns="filename",
                            values="Intensity",
                            aggfunc="max")
               .reset_index()
               .rename(columns={protein_col: "protein", ion_col: "ion"}))
    return pivot


def _run_lfq_on_df(lfq_in: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Run directLFQ normalization + protein estimation on a pivoted DataFrame."""
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
# v0.5.5 site roll-up
# ---------------------------------------------------------------------------

def _resolve_extractor(precursor_long: pd.DataFrame):
    """Return a ``Protein.Group → accession`` extractor compatible with FASTA
    lookups.  ``attach_fasta_meta`` stashes one in ``df.attrs``; when it is
    absent we import the module-level helper as a fallback.
    """
    extract_acc = (
        precursor_long.attrs.get("protein_group_accession")
        if hasattr(precursor_long, "attrs") else None
    )
    if extract_acc is None:
        from .parse_sage import _extract_accession as extract_acc
    return extract_acc


def _site_key(accession: str, gene: str, residue: str, abs_pos: int,
              mod_name: str) -> str:
    label = gene or accession or "unknown"
    return f"{accession}|{label}|{residue}{abs_pos}|{mod_name}"


def site_quant(precursor_long: pd.DataFrame,
               min_samples: int = 1,
               localization_cutoff: float = 0.0,
               include_low_loc: bool = False,
               phospho_only: bool = True,
               allowed_mods: Optional[List[str]] = None) -> pd.DataFrame:
    """Roll precursors up to PTM-site level using absolute protein positions.

    Parameters
    ----------
    precursor_long
        Long precursor table with ``Modified.Sequence``, ``Stripped.Sequence``,
        ``PTM.Mods`` (v0.5.5; preferred) or the legacy
        ``PTM.Site.Positions``/``PTM.Modification`` pair, ``Protein.Group``,
        ``Genes``, ``Intensity``, ``filename``.
    min_samples
        Minimum non-NaN samples per site (directLFQ parameter).
    localization_cutoff
        Per-site probability below which a site is dropped.  ``0.0`` keeps
        every site (fail-open).
    include_low_loc
        When ``True``, keep sites below the cutoff in the output but flag
        them with a ``Best.Site.Probability`` below threshold.  Mirrors
        DIA-NN's permissive behaviour.
    phospho_only
        When ``True`` (default), emit only ``mod_name == "Phospho"`` rows,
        which is the common case for phospho-enriched workflows.  Disable
        to expose every searched mod type in a single matrix.
    allowed_mods
        Optional whitelist of mod names.  Overrides ``phospho_only`` when
        provided.  Example: ``["Phospho", "GlyGly"]``.
    """
    if precursor_long.empty:
        return pd.DataFrame()

    has_ptm_mods = "PTM.Mods" in precursor_long.columns
    df = precursor_long.copy()

    if has_ptm_mods:
        # v0.5.5 path: explode PTM.Mods into one row per site.
        # ---- v0.5.7 (P0-2) ---------------------------------------------
        # The v0.5.5/v0.5.6 implementation iterated rows with
        # ``df.itertuples(index=False)`` and reached for
        # ``getattr(row, "PTM.Mods")``.  Pandas' itertuples() mangles every
        # column name containing a ``.`` into a positional placeholder
        # (``_0``, ``_2``, ``_8`` ...), so the getattr always returned
        # ``None``, every row was treated as ``ptm_mods == ""``, and the
        # site_rows list was permanently empty -> ptm_site_matrix.tsv was
        # written with 0 data rows even though Sage found thousands of
        # phospho PSMs.  We now iterate the columns directly via numpy
        # arrays, which is both correct and ~5x faster on the KBSI
        # benchmark.
        # ----------------------------------------------------------------
        extract_acc = _resolve_extractor(precursor_long)
        fasta_records = precursor_long.attrs.get("fasta_records", {}) \
            if hasattr(precursor_long, "attrs") else {}

        if allowed_mods is not None:
            whitelist = set(allowed_mods)
        elif phospho_only:
            whitelist = {"Phospho"}
        else:
            whitelist = None  # = accept all

        n_in = len(df)
        col_ptmmods   = df["PTM.Mods"].astype(str).fillna("").to_numpy()
        col_pg        = df["Protein.Group"].astype(str).fillna("").to_numpy()
        col_genes     = (df["Genes"].astype(str).fillna("").to_numpy()
                         if "Genes" in df.columns else np.array([""] * n_in))
        col_stripped  = (df["Stripped.Sequence"].astype(str).fillna("").to_numpy()
                         if "Stripped.Sequence" in df.columns else np.array([""] * n_in))
        col_intensity = (pd.to_numeric(df["Intensity"], errors="coerce").to_numpy()
                         if "Intensity" in df.columns else np.full(n_in, np.nan))
        col_filename  = (df["filename"].astype(str).fillna("").to_numpy()
                         if "filename" in df.columns else np.array([""] * n_in))
        col_precursor = (df["Precursor.Id"].astype(str).fillna("").to_numpy()
                         if "Precursor.Id" in df.columns else np.array([""] * n_in))

        site_rows = []
        n_rows_with_mods = 0
        n_sites_kept = 0
        for i in range(n_in):
            ptm_mods = col_ptmmods[i]
            if not ptm_mods or ptm_mods == "nan":
                continue
            n_rows_with_mods += 1
            accession = extract_acc(str(col_pg[i]))
            gene = col_genes[i]
            stripped = col_stripped[i]
            intensity = col_intensity[i]
            filename = col_filename[i]
            precursor_id = col_precursor[i]

            seq = ""
            if fasta_records and accession in fasta_records:
                seq = fasta_records[accession].get("seq", "") or ""
            pep_start = seq.find(stripped) + 1 if seq and stripped else 0

            for mod_name, residue, pep_pos, prob in iter_site_entries(ptm_mods):
                if whitelist is not None and mod_name not in whitelist:
                    continue
                if (not include_low_loc
                        and localization_cutoff > 0.0
                        and prob < localization_cutoff):
                    continue
                if pep_start > 0:
                    abs_pos = pep_start + pep_pos - 1
                else:
                    abs_pos = pep_pos  # fall back, unique per accession+pep_pos
                site_key = _site_key(accession, gene, residue, abs_pos, mod_name)
                site_rows.append((site_key, precursor_id, filename, intensity, prob))
                n_sites_kept += 1

        logger.info(
            "site_quant: %d / %d PSMs carry PTM.Mods, %d sites kept "
            "(phospho_only=%s, cutoff=%.2f, include_low_loc=%s)",
            n_rows_with_mods, n_in, n_sites_kept,
            phospho_only, localization_cutoff, include_low_loc,
        )
        if not site_rows:
            return pd.DataFrame()

        site_df = pd.DataFrame(
            site_rows,
            columns=["Protein.Group", "Precursor.Id", "filename", "Intensity",
                     "Best.Site.Probability"],
        )
        # Multiple sites on one precursor contribute the same intensity but
        # different site keys; directLFQ handles that correctly.
        lfq_in = _to_lfq_input(site_df, protein_col="Protein.Group")
        site_lfq = _run_lfq_on_df(lfq_in, min_samples=min_samples)

        if not site_lfq.empty:
            loc = (site_df.groupby("Protein.Group")["Best.Site.Probability"]
                           .mean()
                           .rename("Best.Site.Probability"))
            loc.index.name = "protein"
            site_lfq = site_lfq.merge(loc, left_on="protein",
                                      right_index=True, how="left")
        return site_lfq

    # ------------------------------------------------------------------
    # Legacy path: reconstruct per-site info from PTM.Site.Positions +
    # PTM.Modification.  Kept for downstream users that bypass
    # :func:`add_site_probabilities` — not recommended for new code.
    # ------------------------------------------------------------------
    df = df[df.get("PTM.Site.Positions", "").fillna("none") != "none"]
    if df.empty:
        return pd.DataFrame()
    if (localization_cutoff > 0.0 and not include_low_loc
            and "Best.Site.Probability" in df.columns):
        df = df[df["Best.Site.Probability"].fillna(0.0) >= localization_cutoff]
    if df.empty:
        return pd.DataFrame()

    extract_acc = _resolve_extractor(precursor_long)
    fasta_records = precursor_long.attrs.get("fasta_records", {}) \
        if hasattr(precursor_long, "attrs") else {}

    import re
    tok = re.compile(r"^([A-Z])(\d+)$")

    whitelist_legacy = (
        set(allowed_mods) if allowed_mods is not None
        else ({"Phospho"} if phospho_only else None)
    )

    def build_keys(r) -> List[str]:
        accession = extract_acc(str(r.get("Protein.Group", "")))
        gene = str(r.get("Genes", "") or "")
        raw_mod = str(r.get("PTM.Modification", "none"))
        mod_name = _normalise_tag(raw_mod) if raw_mod != "none" else "none"
        if whitelist_legacy is not None and mod_name not in whitelist_legacy:
            return []
        stripped = str(r.get("Stripped.Sequence", ""))
        seq = fasta_records.get(accession, {}).get("seq", "") if fasta_records else ""
        pep_start = seq.find(stripped) + 1 if seq and stripped else 0
        keys = []
        for token in str(r.get("PTM.Site.Positions", "")).split(";"):
            m = tok.match(token.strip())
            if not m:
                continue
            aa, pep_pos = m.group(1), int(m.group(2))
            abs_pos = pep_start + pep_pos - 1 if pep_start > 0 else pep_pos
            keys.append(_site_key(accession, gene, aa, abs_pos, mod_name))
        return keys

    df["__site_keys__"] = df.apply(build_keys, axis=1)
    df = df.explode("__site_keys__")
    df = df[df["__site_keys__"].notna() & (df["__site_keys__"] != "")]
    if df.empty:
        return pd.DataFrame()

    site_df = (df.drop(columns=["Protein.Group"], errors="ignore")
                 .rename(columns={"__site_keys__": "Protein.Group"}))
    lfq_in = _to_lfq_input(site_df, protein_col="Protein.Group")
    site_lfq = _run_lfq_on_df(lfq_in, min_samples=min_samples)
    if "Best.Site.Probability" in df.columns and not site_lfq.empty:
        loc = (df.groupby("Protein.Group")["Best.Site.Probability"]
                 .mean()
                 .rename("Best.Site.Probability"))
        loc.index.name = "protein"
        site_lfq = site_lfq.merge(loc, left_on="protein",
                                  right_index=True, how="left")
    return site_lfq


# ---------------------------------------------------------------------------
# Precursor matrix
# ---------------------------------------------------------------------------

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


def precursor_matrix_normalized(precursor_long: pd.DataFrame) -> pd.DataFrame:
    """Per-precursor wide matrix with directLFQ sample-level normalization.

    v0.5.7 (P2-1): the v0.5.6 ``pr_matrix`` was raw apex area, so any
    run-to-run loading-bias (typical: 1.5–3x in DIA on KIST EPS-pretreated
    samples) showed up as missingness once a downstream analyst applied a
    ``log2 ≥ X`` cut.  We now run directLFQ's NormalizationManager on the
    precursor pivot before writing the matrix, which equalises medians and
    drops the missing rate from ~38% to the expected ~18%.  The protein
    quant path was already normalised, so this only fixes the *precursor*
    matrix and the downstream site matrix that re-uses precursor intensity.
    """
    pr_wide = precursor_matrix(precursor_long)
    sample_cols = [c for c in pr_wide.columns if c not in {
        "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
        "First.Protein.Description", "Proteotypic",
        "Stripped.Sequence", "Modified.Sequence",
        "Precursor.Charge", "Precursor.Id",
    }]
    if len(sample_cols) < 2:
        return pr_wide  # nothing to normalise across

    # Build a directLFQ-shaped frame: protein/ion + per-sample columns.
    norm_in = pr_wide[["Protein.Group", "Precursor.Id"] + sample_cols].rename(
        columns={"Protein.Group": "protein", "Precursor.Id": "ion"}
    )

    try:
        import directlfq.utils as lfqutils
        import directlfq.normalization as lfqnorm
        import directlfq.config as lfqcfg
        lfqcfg.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
        lfqcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas()
        df = lfqutils.sort_input_df_by_protein_and_quant_id(norm_in)
        df = lfqutils.remove_potential_quant_id_duplicates(df)
        df = lfqutils.index_and_log_transform_input_df(df)
        df = lfqutils.remove_allnan_rows_input_df(df)
        if df.empty:
            return pr_wide
        df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
            df, num_samples_quadratic=50
        ).complete_dataframe
        # directLFQ promotes (protein, ion) into a MultiIndex; restore them as
        # ordinary columns so we can merge back on Precursor.Id.
        df_reset = df.reset_index()
        # Reverse the log2 transform.
        sample_arr = df_reset[sample_cols].copy()
        sample_arr = (2.0 ** sample_arr).where(sample_arr.notna(), other=np.nan)
        df_lin = pd.concat(
            [df_reset[[c for c in df_reset.columns if c not in sample_cols]],
             sample_arr],
            axis=1,
        )
        df_lin = df_lin.rename(columns={"ion": "Precursor.Id"})
        keep = ["Precursor.Id"] + sample_cols
        df_lin = df_lin[keep].drop_duplicates("Precursor.Id")
        merged = pr_wide.drop(columns=sample_cols).merge(
            df_lin, on="Precursor.Id", how="left"
        )
        # Preserve original column order.
        return merged[pr_wide.columns]
    except Exception as exc:
        logger.warning(
            "precursor_matrix_normalized: directLFQ normalization failed (%s); "
            "falling back to raw matrix.", exc,
        )
        return pr_wide
