"""Convert AlphaDIA precursor output (``precursor.tsv`` / ``precursor.parquet``)
into the long PSM table that ``quantify``, ``mbr`` and ``writer`` consume.

This module is the v0.6.0 counterpart of :mod:`parse_sage`.  Where Sage emits
``results.sage.tsv`` keyed on the engine's own column names, AlphaDIA emits a
publishing-friendly schema documented in ``alphadia/constants/keys.py``:

* ``precursor.sequence``  — bare amino-acid sequence
* ``precursor.mods``      — semicolon-separated mod **names** (or
                            ``Name@Target`` tokens, depending on whether the
                            library was custom-built); empty string for
                            unmodified peptides
* ``precursor.mod_sites`` — semicolon-separated 1-based residue positions
                            (0 = N-term, -1 = C-term)
* ``precursor.charge``    — int
* ``precursor.qval``      — precursor-level q-value
* ``precursor.intensity`` — LFQ precursor intensity
* ``precursor.rt.observed`` — observed RT (seconds)
* ``pg.proteins`` / ``pg.genes`` / ``pg.qval`` — protein-group metadata
* ``run``                 — raw filename (without extension)

The DIA-NN-style ``Modified.Sequence`` column is reconstructed via
:func:`diaquant.modifications.format_diann_sequence` so the rest of the
pipeline (which already speaks DIA-NN syntax courtesy of ``parse_sage``) can
consume the table without modification.

User requirement (v0.6.0): the resulting ``Modified.Sequence`` MUST use the
canonical DIA-NN ``(UniMod:N)`` token (e.g. ``_AAS(UniMod:21)PEPTIDER_``) so
that the precursor matrix (``pr_matrix.tsv``) is interchangeable with DIA-NN
output for any downstream tool that expects that format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .modifications import format_diann_sequence
from .ptm_localization import MOD_RE, add_site_probabilities

_LOG = logging.getLogger(__name__)


# Canonical AlphaDIA -> PTMQuant column rename.  Only the columns that
# downstream code actually needs are renamed; everything else is kept as-is so
# diagnostic dumps still show the AlphaDIA-native columns for debugging.
ALPHADIA_TO_DIANN = {
    "run":                  "filename",
    "precursor.sequence":   "Stripped.Sequence",
    "precursor.charge":     "Precursor.Charge",
    "precursor.qval":       "Peptide.Q.Value",   # AlphaDIA's fdr.fdr cutoff
    "precursor.proba":      "Score",             # classifier probability
    "precursor.intensity":  "Intensity",
    "precursor.rt.observed": "RT",               # seconds; converted below
    "precursor.rt.library": "Predicted.RT",      # library RT (seconds)
    "precursor.mz_observed": "Exp.Mass",
    "precursor.mz_calibrated": "Calc.Mass",
    "pg.name":              "Protein.Group",
    "pg.proteins":          "Protein.Ids",
    "pg.genes":             "Genes",
    "pg.qval":              "Protein.Q.Value",
}


def _strip_mods(seq: str) -> str:
    """Drop any ``(UniMod:N)`` / ``[+xx.xxxx]`` markup from a Modified.Sequence."""
    if not isinstance(seq, str):
        return ""
    bare = MOD_RE.sub("", seq)
    return bare.strip("_")


def _read_alphadia_precursor(path: Path) -> pd.DataFrame:
    """Read either ``precursor.tsv`` or ``precursor.parquet``.

    AlphaDIA's ``search_output.file_format`` config knob decides which one is
    written; we accept both so callers don't need to know.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AlphaDIA precursor file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t", low_memory=False)
    elif suffix == ".csv":
        return pd.read_csv(p, low_memory=False)
    else:
        # Try parquet first (it's the AlphaDIA default), then fall back to TSV.
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p, sep="\t", low_memory=False)


def _build_modified_sequence(df: pd.DataFrame) -> pd.Series:
    """Vectorised wrapper around :func:`format_diann_sequence`.

    AlphaDIA stores mods/mod_sites as semicolon-separated strings, possibly
    NaN when the peptide is unmodified.  We coerce NaN -> ``""`` and call the
    formatter row-by-row.  The column count is bounded (~10^5 rows for a
    typical KBSI run) so a Python-level loop is acceptable here; if profiling
    shows this is a hot spot we can switch to a C extension.
    """
    seqs = df["precursor.sequence"].astype(str).fillna("")
    mods = df["precursor.mods"].astype(str).fillna("").replace("nan", "")
    sites = df["precursor.mod_sites"].astype(str).fillna("").replace("nan", "")
    return pd.Series(
        [format_diann_sequence(s, m, ss) for s, m, ss in zip(seqs, mods, sites)],
        index=df.index,
        name="Modified.Sequence",
    )


def parse_alphadia_precursor(path: Path,
                             site_cutoff: float = 0.75,
                             peptide_fdr: Optional[float] = None,
                             return_unfiltered: bool = False):
    """Read an AlphaDIA precursor table and return the long DIA-NN-style PSM df.

    Parameters
    ----------
    path
        Path to ``precursor.tsv`` or ``precursor.parquet``.
    site_cutoff
        PTM site-probability cutoff for :func:`add_site_probabilities`.
    peptide_fdr
        If not ``None``, additionally filter ``Peptide.Q.Value <= peptide_fdr``.
        Defaults to ``None`` because AlphaDIA already applies its own
        ``fdr.fdr`` cutoff (per-pass) before writing the precursor table, so
        a second filter would be redundant for typical runs.  Set explicitly
        when stacking PTMQuant's own pass-FDR policy on top.
    return_unfiltered
        When True, returns ``(df_filtered, df_full)``.  ``df_full`` carries
        the same columns but skips the FDR filter -- it's the donor pool for
        :mod:`diaquant.mbr`.

    Returns
    -------
    pd.DataFrame
        Long PSM table with the standard PTMQuant columns:
        ``filename, Modified.Sequence, Stripped.Sequence, Precursor.Charge,
        Precursor.Id, Protein.Group, Protein.Ids, Protein.Names, Genes,
        First.Protein.Description, Proteotypic, RT, Predicted.RT,
        Peptide.Q.Value, Protein.Q.Value, Intensity, Score`` plus the PTM
        localisation columns added by :func:`add_site_probabilities`.
    """
    raw = _read_alphadia_precursor(Path(path))
    n_in = len(raw)
    _LOG.info("alphadia precursor table loaded: %d rows from %s", n_in, path)

    # Defensive: AlphaDIA used to ship some columns under slightly different
    # names in older releases; we coerce a couple of well-known aliases here.
    aliases = {
        "raw_name": "run",
        "filename": "run",
        "qval": "precursor.qval",
        "proba": "precursor.proba",
    }
    for old, new in aliases.items():
        if old in raw.columns and new not in raw.columns:
            raw = raw.rename(columns={old: new})

    required = {"precursor.sequence", "precursor.mods", "precursor.mod_sites",
                "precursor.charge", "run"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"AlphaDIA precursor file {path} is missing required columns: "
            f"{sorted(missing)}; got {list(raw.columns)[:20]}..."
        )

    # 1) Reconstruct the DIA-NN Modified.Sequence (UniMod tokens) FIRST so it
    #    carries through the rename + downstream localisation.
    raw["Modified.Sequence"] = _build_modified_sequence(raw)

    # 2) Apply the canonical column renames.
    df = raw.rename(columns={k: v for k, v in ALPHADIA_TO_DIANN.items()
                              if k in raw.columns}).copy()

    # 3) Convert RT seconds -> minutes for parity with parse_sage.
    for col in ("RT", "Predicted.RT"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 60.0

    # 4) Backfill the standard metadata columns parse_sage emits.
    if "Stripped.Sequence" not in df.columns:
        df["Stripped.Sequence"] = df["Modified.Sequence"].astype(str).map(_strip_mods)
    if "Protein.Group" not in df.columns and "Protein.Ids" in df.columns:
        df["Protein.Group"] = df["Protein.Ids"].astype(str).str.split(";").str[0]
    if "Protein.Group" not in df.columns:
        df["Protein.Group"] = ""
    if "Protein.Ids" not in df.columns:
        df["Protein.Ids"] = df["Protein.Group"]
    if "Protein.Names" not in df.columns:
        df["Protein.Names"] = df["Protein.Group"]
    if "Genes" not in df.columns:
        df["Genes"] = df["Protein.Group"]
    if "First.Protein.Description" not in df.columns:
        df["First.Protein.Description"] = ""
    df["Proteotypic"] = (df["Protein.Ids"].astype(str)
                         .str.contains(";").map(lambda b: 0 if b else 1))
    df["Precursor.Id"] = (df["Modified.Sequence"].astype(str)
                          + df["Precursor.Charge"].astype(str))

    # 5) Drop AlphaDIA decoys defensively.  AlphaDIA filters them internally
    #    before writing precursor.tsv, but we re-check via the protein-group
    #    name prefix so the donor pool can never carry a decoy through MBR.
    n_before_decoy = len(df)
    decoy_mask = (df["Protein.Group"].astype(str)
                    .str.startswith(("rev_", "REV_", "DECOY_", "decoy_")))
    n_decoy = int(decoy_mask.sum())
    if n_decoy:
        _LOG.info("alphadia decoy filter: dropped %d / %d PSMs",
                  n_decoy, n_before_decoy)
    df = df[~decoy_mask].copy()
    df_full = df.copy()

    # 6) Optional second-pass FDR filter.
    if peptide_fdr is not None and "Peptide.Q.Value" in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df["Peptide.Q.Value"], errors="coerce")
                  .fillna(1.0) <= peptide_fdr]
        _LOG.info(
            "alphadia FDR filter (peptide_q \u2264 %.3f): %d \u2192 %d PSMs",
            peptide_fdr, before, len(df),
        )

    # 7) PTM site localisation.  AlphaDIA's MS2 scoring already produces a
    #    well-localised result (the ML scorer punishes ambiguous placements),
    #    but we still run the same add_site_probabilities pass that parse_sage
    #    uses so the downstream site-quant code finds identical column names.
    df = add_site_probabilities(
        df.rename(columns={"Modified.Sequence": "peptide",
                           "Precursor.Charge": "charge"}),
        cutoff=site_cutoff,
    )
    df = df.rename(columns={"peptide": "Modified.Sequence",
                            "charge": "Precursor.Charge"})

    if return_unfiltered:
        df_full = add_site_probabilities(
            df_full.rename(columns={"Modified.Sequence": "peptide",
                                    "Precursor.Charge": "charge"}),
            cutoff=0.0,
        )
        df_full = df_full.rename(columns={"peptide": "Modified.Sequence",
                                          "charge": "Precursor.Charge"})
        return df, df_full
    return df


# ---- FASTA metadata join (re-exports parse_sage's helpers so callers don't
# need to import two parsers).
from .parse_sage import attach_fasta_meta, load_fasta_records  # noqa: E402,F401
