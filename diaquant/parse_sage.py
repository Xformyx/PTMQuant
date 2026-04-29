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
            "FDR filter (peptide_q \u2264 %.3f): %d \u2192 %d PSMs", peptide_fdr, before, len(df)
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


def load_fasta_records(fasta: Path) -> dict:
    """Return ``{accession: {"name", "gene", "descr", "seq"}}`` for every entry.

    Exposed as a module-level helper so the site-matrix roll-up (new in 0.5.3)
    can look up the parent protein sequence to compute absolute site positions
    without re-parsing the FASTA.
    """
    from pyteomics import fasta as pf

    records: dict = {}
    with pf.read(str(fasta)) as reader:
        for header, seq in reader:
            # UniProt-style:  >sp|P12345|GENE_HUMAN Description OS=... GN=...
            m = re.match(r"^[a-z]{2}\|([^|]+)\|(\S+)\s*(.*)$", header)
            if not m:
                acc = header.split()[0]
                records[acc] = {
                    "name": acc, "gene": "", "descr": header, "seq": seq,
                }
                continue
            acc, name, rest = m.groups()
            gene = ""
            gn = re.search(r"GN=(\S+)", rest)
            if gn:
                gene = gn.group(1)
            descr = re.sub(r"\s+(OS|OX|GN|PE|SV)=.*$", "", rest)
            records[acc] = {
                "name": name, "gene": gene, "descr": descr, "seq": seq,
            }
    return records


_PROTEIN_GROUP_ACC_RE = re.compile(r"^(?:[a-z]{2}\|)?([A-Za-z0-9._-]+)")


def _extract_accession(protein_group: str) -> str:
    """Return the bare UniProt accession from a ``Protein.Group`` cell.

    Sage emits ``sp|P12345|GENE_HUMAN`` style identifiers in ``Protein.Group``,
    while :func:`load_fasta_records` keys its dict on the bare accession
    (``P12345``).  v0.5.3 (and v0.5.2) accidentally looked up the full
    ``sp|...|...`` string against that dict and silently filled every Genes /
    Protein.Names / First.Protein.Description cell with an empty string
    — and the v0.5.3 site-matrix roll-up additionally lost every row because
    ``fasta_records[Protein.Group]`` always missed.  This helper normalises
    the lookup key so both the precursor matrix metadata join *and* the site
    matrix absolute-position resolver work for UniProt-style FASTAs.

    Decoy entries are emitted by Sage as ``rev_sp|P12345|GENE_HUMAN``; the
    leading ``rev_`` is stripped so decoys still join to their target FASTA
    record (callers usually drop decoys upstream, but we keep the join lossless
    in case they don't).
    """
    if not isinstance(protein_group, str) or not protein_group:
        return ""
    pg = protein_group[4:] if protein_group.startswith("rev_") else protein_group
    # Protein.Group can carry multiple semicolon-delimited accessions for
    # protein groups; we use the first one as the representative for metadata.
    pg = pg.split(";")[0]
    m = _PROTEIN_GROUP_ACC_RE.match(pg)
    return m.group(1) if m else pg


def attach_fasta_meta(df: pd.DataFrame, fasta: Path) -> pd.DataFrame:
    """Fill Genes / Protein.Names / First.Protein.Description from a UniProt FASTA.

    The parsed FASTA records are cached on ``df.attrs['fasta_records']`` so
    the 0.5.3 site-matrix roll-up can compute absolute PTM positions without
    re-reading the FASTA.
    """
    records = load_fasta_records(fasta)

    # Vectorised accession extraction — the per-row apply() in v0.5.3 not only
    # masked the lookup-key bug but was also a perf hot spot on large pr_matrix.
    accessions = df["Protein.Group"].map(_extract_accession)

    def _col(field: str) -> pd.Series:
        return accessions.map(lambda a: records.get(a, {}).get(field, ""))

    df["Protein.Names"] = _col("name")
    df["Genes"] = _col("gene")
    df["First.Protein.Description"] = _col("descr")
    # Stash records *and* the accession map so site_quant() can look up
    # sequences without re-parsing the FASTA *and* without re-parsing the
    # Protein.Group string for every site.
    df.attrs["fasta_records"] = records
    df.attrs["protein_group_accession"] = _extract_accession
    return df
