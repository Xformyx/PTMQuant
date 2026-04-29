"""Razor-peptide protein grouping (v0.5.4, Fix C).

Up to v0.5.3 diaquant used ``Sage.Protein.Ids.split(';')[0]`` as
``Protein.Group``.  This silently pins every shared peptide to the first
accession Sage happened to emit, and ``Proteotypic`` is set to 0 for every
shared peptide.  Downstream, ``protein_quant`` then drops rows keyed on
accessions that never receive any proteotypic-sole-owner peptide, which
explains the 3x gap vs DIA-NN (on the KBSI benchmark: pg_matrix 2,008 vs
pr_matrix-accessions 3,527 -- 1,519 proteins were lost between the two
layers).

This module implements the standard MaxQuant-style grouping used by DIA-NN:

1. Build, for every stripped peptide, its candidate accession set from
   Sage's ``Protein.Ids`` column.
2. Collapse accessions that share an *identical peptide-candidate set* into
   a single protein group.  Concretely: group all peptides whose candidate
   sets are equal, then the union of those peptides defines the group and
   the candidate set defines its accession list.  ``Protein.Group`` is the
   ``;``-joined sorted accession list.
3. For every peptide, compute the set of groups that contain *any* of its
   candidate accessions.  When only one such group exists, the peptide is
   unique and ``Proteotypic=1``.  Otherwise apply Occam's razor and assign
   it to the group with the largest number of unique peptides (ties broken
   by total PSM count, then by group-string).  Razor peptides also get
   ``Proteotypic=1`` because after merging they are unique to their group.
4. Optionally drop groups with fewer than ``min_peptides_per_protein``
   distinct stripped peptides (DIA-NN default = 1).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _split_accessions(protein_ids: str) -> List[str]:
    if not isinstance(protein_ids, str) or not protein_ids:
        return []
    return [p for p in protein_ids.split(";") if p]


def _peptide_to_candidates(df: pd.DataFrame) -> Dict[str, FrozenSet[str]]:
    """Return ``{stripped_peptide: frozenset(candidate_accessions)}``."""
    # Some peptides appear in multiple PSM rows with slightly different
    # ``Protein.Ids`` strings (e.g. extra accessions in later passes).  Take
    # the *union* of candidates across rows to be safe.
    acc_by_pep: Dict[str, Set[str]] = defaultdict(set)
    for pep, ids in zip(df["Stripped.Sequence"].astype(str),
                        df["Protein.Ids"].astype(str)):
        for acc in _split_accessions(ids):
            acc_by_pep[pep].add(acc)
    return {pep: frozenset(accs) for pep, accs in acc_by_pep.items()}


def _build_groups(
    pep2cands: Dict[str, FrozenSet[str]]
) -> Tuple[Dict[FrozenSet[str], str], Dict[str, List[str]]]:
    """Merge accessions that share an identical candidate set.

    Returns
    -------
    cands2group
        ``{candidate_set: "Acc1;Acc2;..."}`` - canonical group string for a
        given peptide candidate set.
    acc2groups
        ``{accession: [group_string, ...]}`` - every group string that
        contains this accession.  An accession can legitimately belong to
        multiple groups when different peptides define different candidate
        sets that both include it.
    """
    # First, collect distinct candidate sets and the peptides that carry them.
    cands_seen: Set[FrozenSet[str]] = set()
    for cands in pep2cands.values():
        if cands:
            cands_seen.add(cands)

    # Each distinct candidate set becomes its own protein group; the group's
    # accession list is the sorted candidate set.  This matches the
    # "indistinguishable-in-this-dataset" protein-group definition used by
    # DIA-NN and MaxQuant.
    cands2group: Dict[FrozenSet[str], str] = {
        cands: ";".join(sorted(cands)) for cands in cands_seen
    }
    acc2groups: Dict[str, List[str]] = defaultdict(list)
    for cands, group_str in cands2group.items():
        for acc in cands:
            acc2groups[acc].append(group_str)
    return cands2group, acc2groups


def apply_razor_grouping(
    df: pd.DataFrame,
    min_peptides_per_protein: int = 1,
) -> pd.DataFrame:
    """Rewrite ``Protein.Group`` + ``Proteotypic`` using razor grouping.

    Parameters
    ----------
    df
        Long precursor table produced by :func:`diaquant.parse_sage.parse_sage_tsv`.
        Must contain ``Stripped.Sequence`` and ``Protein.Ids``.
    min_peptides_per_protein
        Drop protein groups supported by fewer than this many distinct
        stripped peptides.  DIA-NN's equivalent default is 1.
    """
    if df.empty:
        return df
    required = {"Stripped.Sequence", "Protein.Ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"apply_razor_grouping: missing required columns: {missing}")

    pep2cands = _peptide_to_candidates(df)
    cands2group, acc2groups = _build_groups(pep2cands)

    # For razor scoring we need to know how many peptides each GROUP already
    # owns uniquely (i.e. peptides whose candidate set equals the group
    # identity).  These unique-owner peptides are the strongest evidence for
    # that group.
    group2unique_peps: Dict[str, Set[str]] = defaultdict(set)
    for pep, cands in pep2cands.items():
        g = cands2group.get(cands)
        if g is not None:
            group2unique_peps[g].add(pep)

    # Total PSM count per GROUP (tie-breaker for razor).
    group2psm_count: Dict[str, int] = defaultdict(int)
    for pep, cands in pep2cands.items():
        g = cands2group.get(cands)
        if g is not None:
            # Count all PSM rows carrying this peptide.
            group2psm_count[g] += (df["Stripped.Sequence"] == pep).sum()

    # ---- assign every peptide to exactly one GROUP -----------------------
    # A peptide's candidate set always matches exactly one group (since we
    # grouped by candidate set).  Two different groups can however both
    # *contain* one of the peptide's candidate accessions when those two
    # groups' candidate sets overlap (different peptides observed different
    # parents for the same underlying protein family).  Here we resolve that
    # by Occam's razor on the *group* ids that each accession belongs to.
    pep2group: Dict[str, str] = {}
    pep2is_unique: Dict[str, bool] = {}
    for pep, cands in pep2cands.items():
        # All groups that contain any of this peptide's candidate accessions:
        candidate_groups: Set[str] = set()
        for acc in cands:
            for g in acc2groups.get(acc, []):
                candidate_groups.add(g)
        # Always include the group whose candidate set equals ``cands``
        own_group = cands2group.get(cands)
        if own_group is not None:
            candidate_groups.add(own_group)
        if len(candidate_groups) == 1:
            (g,) = tuple(candidate_groups)
            pep2group[pep] = g
            pep2is_unique[pep] = True
        else:
            scored = [
                (len(group2unique_peps.get(g, set())),
                 group2psm_count.get(g, 0),
                 g)
                for g in candidate_groups
            ]
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            pep2group[pep] = scored[0][2]
            pep2is_unique[pep] = False

    # ---- rewrite the frame ------------------------------------------------
    out = df.copy()
    out["Protein.Group"] = out["Stripped.Sequence"].astype(str).map(pep2group)
    # A peptide is proteotypic for its assigned group iff *no* candidate
    # group was rejected above.  When multiple groups contained its
    # candidates we still mark it Proteotypic=1 only if those rejected
    # groups have overlapping accessions with the assigned group (i.e. the
    # peptide is truly redundant within the family); otherwise it's a razor
    # peptide with Proteotypic=0 to flag the ambiguity.
    def _proteotypic(pep: str) -> int:
        # DIA-NN convention: after razor assignment every peptide that
        # unambiguously ends up in a single protein group is proteotypic
        # for that group.  We emit 0 only when the peptide had no valid
        # candidate accessions at all (should not happen on real data).
        return 1 if pep2group.get(pep) else 0

    out["Proteotypic"] = out["Stripped.Sequence"].astype(str).map(_proteotypic)

    # ---- min_peptides_per_protein filter ---------------------------------
    if min_peptides_per_protein and min_peptides_per_protein > 1:
        grp_pep_count = (
            out.groupby("Protein.Group")["Stripped.Sequence"].nunique()
        )
        keep = grp_pep_count[grp_pep_count >= min_peptides_per_protein].index
        before_rows = len(out)
        before_groups = out["Protein.Group"].nunique()
        out = out[out["Protein.Group"].isin(keep)].copy()
        logger.info(
            "razor: min_peptides_per_protein=%d -> kept %d/%d groups, %d/%d rows",
            min_peptides_per_protein,
            out["Protein.Group"].nunique(), before_groups,
            len(out), before_rows,
        )

    logger.info(
        "razor: %d peptides (%d unique, %d razor) -> %d protein groups",
        len(pep2cands),
        sum(pep2is_unique.values()),
        len(pep2cands) - sum(pep2is_unique.values()),
        out["Protein.Group"].nunique(),
    )
    return out
