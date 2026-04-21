"""Write DIA-NN compatible TSV outputs.

The two flagship files are ``report.pr_matrix.tsv`` (precursor × samples) and
``report.pg_matrix.tsv`` (protein-group × samples).  We replicate the exact
column layout used by DIA-NN 1.9 so that downstream tools (Perseus, R/limma,
PTMsite-Finder, etc.) consume diaquant output without modification.

We additionally emit ``report.ptm_site_matrix.tsv``, which DIA-NN does not
natively provide and which is the central improvement of diaquant: every row
is a (gene, residue, position, PTM type) tuple quantified with directLFQ
across all runs.
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


def write_pr_matrix(pr_wide: pd.DataFrame, out: Path) -> Path:
    """Write a DIA-NN style precursor matrix."""
    sample_cols = [c for c in pr_wide.columns if c not in PR_COLS]
    pr_wide = pr_wide[PR_COLS + sample_cols]
    pr_wide.to_csv(out, sep="\t", index=False, na_rep="")
    return out


def write_pg_matrix(pg_lfq: pd.DataFrame,
                    pr_wide: pd.DataFrame,
                    out: Path) -> Path:
    """Write a DIA-NN style protein-group matrix.

    ``pg_lfq`` is the output of ``directlfq.run_lfq`` (one row per protein,
    one column per sample, plus a ``protein`` index).  We enrich it with the
    metadata columns DIA-NN exposes (gene names, description, sequence count)
    by joining against the precursor table.
    """
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
    """Write a PTM site × sample LFQ matrix.

    The synthetic ``protein`` identifier is split back into ``Genes``,
    ``PTM.Site``, ``PTM.Modification`` columns for readability.
    """
    if site_lfq.empty:
        out.write_text("Genes\tPTM.Site\tPTM.Modification\n")
        return out
    df = site_lfq.copy()
    keys = df["protein"].str.split("_", n=2, expand=True)
    df.insert(0, "Genes", keys[0])
    df.insert(1, "PTM.Site", keys[1])
    df.insert(2, "PTM.Modification", keys[2])
    df = df.drop(columns=["protein"])
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
        "Q.Value", "Protein.Q.Value", "Intensity", "RT", "Predicted.RT",
    ] if c in precursor_long.columns]
    precursor_long[keep].rename(columns={"filename": "File.Name"}) \
                       .to_csv(out, sep="\t", index=False, na_rep="")
    return out
