"""Write DIA-NN compatible TSV outputs.

The two flagship files are ``report.pr_matrix.tsv`` (precursor \u00d7 samples) and
``report.pg_matrix.tsv`` (protein-group \u00d7 samples).  We replicate the exact
column layout used by DIA-NN 1.9 so that downstream tools (Perseus, R/limma,
PTMsite-Finder, etc.) consume diaquant output without modification.

We additionally emit ``report.ptm_site_matrix.tsv``, which DIA-NN does not
natively provide and which is the central improvement of diaquant: every row
is a (protein, gene, residue, absolute position, PTM type) tuple quantified
with directLFQ across all runs.

v0.5.3 changes
--------------
* ``write_site_matrix`` now splits the 4-field synthetic site key
  ``<accession>|<gene>|<AA><abs_pos>|<Mod>`` produced by
  :func:`diaquant.quantify.site_quant` into four readable columns
  (``Protein.Group``, ``Genes``, ``PTM.Site``, ``PTM.Modification``) instead
  of the previous 3-field split that dropped the accession and used
  peptide-local positions.
* A ``Best.Site.Probability`` column is written when the site table carries
  one (populated by ``site_quant`` from the mean localization probability per
  site) so downstream tools can re-filter without re-running diaquant.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# columns used by DIA-NN report.pr_matrix.tsv
PR_COLS = [
    "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
    "First.Protein.Description", "Proteotypic",
    "Stripped.Sequence", "Modified.Sequence",
    "Precursor.Charge", "Precursor.Id",
]

# columns used by DIA-NN report.pg_matrix.tsv
PG_COLS = [
    "Protein.Group", "Protein.Names", "Genes",
    "First.Protein.Description", "N.Sequences", "N.Proteotypic.Sequences",
]

# columns used by the 0.5.3 report.ptm_site_matrix.tsv
SITE_COLS = [
    "Protein.Group", "Genes", "PTM.Site", "PTM.Modification",
    "Best.Site.Probability",
]


def write_pr_matrix(pr_wide: pd.DataFrame, out: Path) -> Path:
    """Write a DIA-NN style precursor matrix."""
    sample_cols = [c for c in pr_wide.columns if c not in PR_COLS]
    pr_wide = pr_wide[PR_COLS + sample_cols]
    pr_wide.to_csv(out, sep="\t", index=False, na_rep="")
    return out


def write_pg_matrix(pg_lfq: pd.DataFrame,
                    pr_wide: pd.DataFrame,
                    out: Path) -> Path:
    """Write a DIA-NN style protein-group matrix."""
    meta = (pr_wide[["Protein.Group", "Protein.Names", "Genes",
                     "First.Protein.Description", "Stripped.Sequence",
                     "Proteotypic"]]
            .drop_duplicates(subset=["Protein.Group", "Stripped.Sequence"]))
    counts = (meta.groupby("Protein.Group")
                  .agg(N_Sequences=("Stripped.Sequence", "nunique"),
                       N_Proteotypic_Sequences=(
                           "Proteotypic",
                           lambda s: int((s.astype(str) == "1").sum()))))
    meta_first = (meta.drop(columns=["Stripped.Sequence", "Proteotypic"])
                      .drop_duplicates("Protein.Group")
                      .merge(counts, on="Protein.Group"))
    meta_first = meta_first.rename(columns={
        "N_Sequences": "N.Sequences",
        "N_Proteotypic_Sequences": "N.Proteotypic.Sequences",
    })

    pg = pg_lfq.rename(columns={"protein": "Protein.Group"})
    out_df = meta_first.merge(pg, on="Protein.Group", how="right")
    sample_cols = [c for c in pg.columns if c != "Protein.Group"]
    out_df = out_df[PG_COLS + sample_cols]
    out_df.to_csv(out, sep="\t", index=False, na_rep="")
    return out


def write_site_matrix(site_lfq: pd.DataFrame, out: Path) -> Path:
    """Write a PTM site \u00d7 sample LFQ matrix.

    The synthetic ``protein`` identifier produced by
    :func:`diaquant.quantify.site_quant` is ``<accession>|<gene>|<AA><abs_pos>|<Mod>``.
    We split it back into ``Protein.Group``, ``Genes``, ``PTM.Site``, and
    ``PTM.Modification`` columns so the file is directly usable in Perseus /
    R / limma without any further wrangling.
    """
    if site_lfq is None or site_lfq.empty:
        out.write_text("\t".join(SITE_COLS) + "\n")
        return out

    df = site_lfq.copy()
    keys = df["protein"].astype(str).str.split("|", n=3, expand=True)
    # Handle legacy 3-field keys (``GENE_S123_Mod``) for backward compatibility.
    if keys.shape[1] < 4:
        legacy = df["protein"].astype(str).str.split("_", n=2, expand=True)
        df.insert(0, "Protein.Group", "")
        df.insert(1, "Genes", legacy[0])
        df.insert(2, "PTM.Site", legacy[1])
        df.insert(3, "PTM.Modification", legacy[2])
    else:
        df.insert(0, "Protein.Group", keys[0])
        df.insert(1, "Genes", keys[1])
        df.insert(2, "PTM.Site", keys[2])
        df.insert(3, "PTM.Modification", keys[3])

    df = df.drop(columns=["protein"])

    # Reorder: identity columns first, loc-prob next (if present), then samples.
    id_cols = ["Protein.Group", "Genes", "PTM.Site", "PTM.Modification"]
    if "Best.Site.Probability" in df.columns:
        id_cols = id_cols + ["Best.Site.Probability"]
    sample_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + sample_cols]
    df.to_csv(out, sep="\t", index=False, na_rep="")
    return out


def write_main_report(precursor_long: pd.DataFrame, out: Path) -> Path:
    """Write a long-format ``report.tsv`` analogous to DIA-NN's main report."""
    keep = [c for c in [
        "filename", "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
        "First.Protein.Description", "Proteotypic", "Stripped.Sequence",
        "Modified.Sequence", "Precursor.Charge", "Precursor.Id",
        "PTM.Site.Positions", "PTM.Modification",
        "Best.Site.Probability", "PTM.Site.Confident",
        "Q.Value", "Protein.Q.Value", "Intensity", "RT", "RT.Aligned", "Predicted.RT",
    ] if c in precursor_long.columns]
    precursor_long[keep].rename(columns={"filename": "File.Name"}) \
                       .to_csv(out, sep="\t", index=False, na_rep="")
    return out
