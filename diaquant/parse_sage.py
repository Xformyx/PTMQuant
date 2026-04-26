"""Convert a ``results.sage.tsv`` table into the long precursor table that
``quantify`` and ``writer`` consume.

We add the standard DIA-NN column names so that downstream code stays
vendor-neutral (e.g. ``Modified.Sequence`` instead of Sage's ``peptide``).
The mapping is intentionally explicit – it's the contract between Sage and
the rest of diaquant.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .ptm_localization import MOD_RE, add_site_probabilities


SAGE_TO_DIANN = {
    "filename":           "filename",
    "scannr":             "Scan.Number",
    "peptide":            "Modified.Sequence",
    "stripped_peptide":   "Stripped.Sequence",
    "proteins":           "Protein.Ids",
    "charge":             "Precursor.Charge",
    "calcmass":           "Calc.Mass",
    "expmass":            "Exp.Mass",
    "rt":                 "RT",
    "predicted_rt":       "Predicted.RT",
    "spectrum_q":         "Q.Value",
    "peptide_q":          "Peptide.Q.Value",
    "protein_q":          "Protein.Q.Value",
    "ms1_intensity":      "MS1.Intensity",
    "ms2_intensity":      "Intensity",          # MS2 area = quant signal
    "hyperscore":         "Score",
}


def _strip_mods(seq: str) -> str:
    return MOD_RE.sub("", seq)


def parse_sage_tsv(path: Path,
                   site_cutoff: float = 0.75,
                   peptide_fdr: float = 0.01) -> pd.DataFrame:
    """Read Sage results, add DIA-NN-style columns and PTM site probabilities."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Force all object-like columns to native Python dtypes so string
    # operations work regardless of whether pandas uses Arrow or numpy backing.
    df = df.copy()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        try:
            df[col] = df[col].astype(str).where(df[col].notna(), other=pd.NA)
        except Exception:
            pass
    df = df.rename(columns=SAGE_TO_DIANN)

    # Sage outputs retention times in seconds; convert to minutes for consistency
    # with DIA-NN-style downstream code (RT alignment, rt_prediction_tolerance_min).
    for col in ("RT", "Predicted.RT"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 60.0

    if "Stripped.Sequence" not in df.columns:
        df["Stripped.Sequence"] = df["Modified.Sequence"].astype(str).map(_strip_mods)

    # primary protein group = first accession in the semicolon list
    df["Protein.Group"] = df["Protein.Ids"].astype(str).str.split(";").str[0]
    df["Protein.Names"] = df["Protein.Group"]
    df["Genes"] = df["Protein.Group"]                     # filled by FASTA-meta join below
    df["First.Protein.Description"] = ""
    df["Proteotypic"] = (df["Protein.Ids"].astype(str)
                         .str.contains(";").map(lambda b: 0 if b else 1))
    df["Precursor.Id"] = (df["Modified.Sequence"].astype(str)
                          + df["Precursor.Charge"].astype(str))

    # filter: target hits at configured peptide FDR
    if "Peptide.Q.Value" in df.columns:
        before = len(df)
        df = df[df["Peptide.Q.Value"] <= peptide_fdr]
        import logging
        logging.getLogger(__name__).info(
            "FDR filter (peptide_q ≤ %.3f): %d → %d PSMs", peptide_fdr, before, len(df)
        )

    # PTM site localization
    df = add_site_probabilities(df.rename(columns={"Modified.Sequence": "peptide",
                                                   "Precursor.Charge": "charge"}),
                                cutoff=site_cutoff)
    df = df.rename(columns={"peptide": "Modified.Sequence",
                            "charge": "Precursor.Charge"})

    # short-hand Modification name = first mod found in the sequence (or 'none')
    def first_mod(seq: str) -> str:
        m = MOD_RE.search(seq)
        return m.group(0).strip("[]()") if m else "none"
    df["PTM.Modification"] = df["Modified.Sequence"].map(first_mod)

    return df


def attach_fasta_meta(df: pd.DataFrame, fasta: Path) -> pd.DataFrame:
    """Fill Genes / Protein.Names / First.Protein.Description from a UniProt FASTA."""
    from pyteomics import fasta as pf

    meta = {}
    with pf.read(str(fasta)) as reader:
        for header, _seq in reader:
            # >sp|P12345|GENE_HUMAN Description OS=… OX=… GN=GeneSym PE=… SV=…
            m = re.match(r"^[a-z]{2}\|([^|]+)\|(\S+)\s*(.*)$", header)
            if not m:
                continue
            acc, name, rest = m.groups()
            gene = ""
            gn = re.search(r"GN=(\S+)", rest)
            if gn:
                gene = gn.group(1)
            descr = re.sub(r"\s+(OS|OX|GN|PE|SV)=.*$", "", rest)
            meta[acc] = (name, gene, descr)

    def fill(row, idx):
        return meta.get(row["Protein.Group"], ("", "", ""))[idx]

    df["Protein.Names"] = df.apply(lambda r: fill(r, 0), axis=1)
    df["Genes"] = df.apply(lambda r: fill(r, 1), axis=1)
    df["First.Protein.Description"] = df.apply(lambda r: fill(r, 2), axis=1)
    return df
