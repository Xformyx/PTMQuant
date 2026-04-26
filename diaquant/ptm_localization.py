"""PTM site localization for DIA-NN-style ``Modified.Sequence`` strings.

Sage already enumerates every variable-mod placement and scores them with the
hyperscore.  For each peptide sequence × charge × precursor m/z we group all
candidate peptidoforms and compute a *site probability* in the spirit of
Ascore / PTMProphet:

    P(site_i) = exp(score_i) / Σ_j exp(score_j)

This is a fast, log-space softmax over the hyperscores returned by Sage and is
sufficient for the first release.  A future version will replace this step with
a Prosit/AlphaPeptDeep-rescored Δ-score for higher accuracy.

Two artefacts are produced for every protein:

* ``site_probability``     ∈ [0,1]   – one value per (peptide, mod_position)
* ``site_pass_cutoff``     bool      – True if probability ≥ user cutoff (default 0.75)

The output is the same Sage table plus three columns: ``Best.Site.Position``,
``Best.Site.Probability`` and ``PTM.Site.Confident``.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

# Matches both DIA-NN and Sage modification syntaxes:
#   AAA[+79.9663]K          (Sage)
#   AAAS(UniMod:21)K        (DIA-NN)
MOD_RE = re.compile(r"\[([+-]?\d+\.\d+)\]|\(UniMod:(\d+)\)|\(([^)]+)\)")


def extract_mod_positions(mod_seq: str) -> List[Tuple[int, str]]:
    """Return list of (1-based position in the bare sequence, mod-name)."""
    positions: List[Tuple[int, str]] = []
    plain_pos = 0
    i = 0
    while i < len(mod_seq):
        ch = mod_seq[i]
        if ch.isalpha():
            plain_pos += 1
            i += 1
            continue
        m = MOD_RE.match(mod_seq, i)
        if m:
            tag = m.group(0).strip("[]()")
            positions.append((plain_pos, tag))
            i = m.end()
            continue
        i += 1
    return positions


def _sage_score_column(psm: pd.DataFrame) -> str:
    """Raw Sage TSV uses ``hyperscore``; after ``parse_sage`` rename it is ``Score``."""
    if "hyperscore" in psm.columns:
        return "hyperscore"
    if "Score" in psm.columns:
        return "Score"
    raise KeyError(
        "Expected Sage match score column 'hyperscore' or 'Score'; "
        f"have: {list(psm.columns)}"
    )


def add_site_probabilities(psm: pd.DataFrame,
                           cutoff: float = 0.75) -> pd.DataFrame:
    """Add Best.Site.Probability and PTM.Site.Confident columns to a Sage PSM table.

    The Sage ``results.sage.tsv`` columns we rely on are:

    * ``peptide``        – modified sequence in Sage syntax
    * ``proteins``       – semicolon-separated UniProt accessions
    * ``hyperscore``     – Sage match score (higher is better); renamed to ``Score`` in diaquant
    * ``filename``       – mzML file that produced the PSM
    * ``charge``         – precursor charge
    """
    if "peptide" not in psm.columns:
        raise KeyError("Expected Sage column 'peptide' in input table.")

    if psm.empty:
        return psm.assign(
            **{
                "Best.Site.Probability": np.nan,
                "PTM.Site.Confident": False,
                "PTM.Site.Positions": "",
            }
        )

    score_col = _sage_score_column(psm)

    # bare sequence (modifications stripped) used to group peptidoforms
    bare = psm["peptide"].str.replace(MOD_RE, "", regex=True)
    psm = psm.copy()
    psm["_bare"] = bare

    # group by file × bare-seq × charge → softmax over hyperscores
    grp = psm.groupby(["filename", "_bare", "charge"])
    psm["Best.Site.Probability"] = grp[score_col].transform(
        lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    )
    psm["PTM.Site.Confident"] = psm["Best.Site.Probability"] >= cutoff

    # attach the literal mod position list as a string for downstream writers
    psm["PTM.Site.Positions"] = psm["peptide"].apply(
        lambda s: ";".join(f"{aa}{pos}" for pos, aa in
                            [(p, t) for p, t in extract_mod_positions(s)])
    )

    return psm.drop(columns=["_bare"])
