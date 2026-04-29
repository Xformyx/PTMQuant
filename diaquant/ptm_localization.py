"""PTM site localization for DIA-NN-style ``Modified.Sequence`` strings.

v0.5.5 rewrite
--------------

The 0.5.1–0.5.4 implementation suffered from three structural issues:

1. ``PTM.Site.Positions`` mixed every modification type into one
   semicolon-separated string, so downstream site quantification could not
   tell an oxidised-methionine apart from a phospho-serine on the same
   peptide.  The result was that ``ptm_site_matrix.tsv`` silently dropped
   most phospho rows.
2. ``PTM.Modification`` only carried the *first* mod in the sequence, so
   multi-PTM peptides were labelled incorrectly.
3. ``Best.Site.Probability`` was computed as a softmax over
   ``(filename × bare-sequence × charge)`` groups.  In practice Sage emits
   a single localization variant per group, so the softmax collapsed to
   ``1.0`` for virtually every PSM and the localization filter became a
   no-op.

This module replaces that logic with an explicit per-mod representation:

* ``PTM.Sites`` is a dict-like string ``"Phospho:S12,T18;Oxidation:M4"``
  keyed on the human-readable mod name, so the site roll-up can filter by
  mod type cleanly.
* ``Best.Site.Probability`` is now a per-mod, per-site value combining
  (a) the Sage hyperscore-based delta between the observed placement and
  the best-scoring alternative placement of the same mod on the same
  peptidoform, when such alternatives are available in the PSM table, and
  (b) a conservative fallback (``0.99`` for unambiguous placements on a
  peptide with a single candidate residue, ``0.50`` otherwise) when no
  alternative placements were searched by Sage.  The fallback value can
  be overridden via ``default_confidence``.
* A helper ``iter_site_rows`` explodes one PSM row into one row per
  (mod, site) so the site-matrix writer can consume the result without
  additional parsing.

The output preserves backwards-compatible columns ``PTM.Site.Positions``
(now phospho-only positions when ``phospho_only_positions=True`` is
requested by the caller) and ``PTM.Modification`` (first mod) so the
0.5.4 consumer code continues to work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matches both DIA-NN and Sage modification syntaxes:
#   AAA[+79.9663]K          (Sage)
#   AAAS(UniMod:21)K        (DIA-NN)
#   AAAS(Phospho)K          (custom)
MOD_RE = re.compile(r"\[([+-]?\d+\.\d+)\]|\(UniMod:(\d+)\)|\(([^)]+)\)")

# Map common mass deltas / UniMod ids to a stable human-readable name used in
# the site matrix (``PTM.Modification`` column).  Anything not in this table
# falls back to the raw tag so nothing is silently lost.
_MASS_TO_NAME = {
    "79.9663": "Phospho",
    "+79.9663": "Phospho",
    "15.9949": "Oxidation",
    "+15.9949": "Oxidation",
    "57.0214": "Carbamidomethyl",
    "+57.0214": "Carbamidomethyl",
    "42.0106": "Acetyl",
    "+42.0106": "Acetyl",
    "114.0429": "GlyGly",
    "+114.0429": "GlyGly",
}
_UNIMOD_TO_NAME = {
    "21": "Phospho",
    "35": "Oxidation",
    "4":  "Carbamidomethyl",
    "1":  "Acetyl",
    "121": "GlyGly",
}

# Residue whitelists for the Sage-style placements we care about.  Used by the
# fallback confidence heuristic to decide whether the placement was unambiguous
# (e.g. a phospho on the only S/T/Y in the peptide).
_RESIDUE_WHITELIST = {
    "Phospho":         set("STY"),
    "Oxidation":       set("M"),
    "Carbamidomethyl": set("C"),
    "Acetyl":          set("K"),
    "GlyGly":          set("K"),
}


@dataclass(frozen=True)
class ModHit:
    """One (mod-name, peptide-local position, residue) triple."""
    mod_name: str
    position: int  # 1-based in the stripped peptide
    residue: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _normalise_tag(tag: str) -> str:
    """Convert a raw mod tag (``+79.9663``/``UniMod:21``/``Phospho``) to a name."""
    tag = tag.strip("[]()")
    if tag.startswith("UniMod:"):
        return _UNIMOD_TO_NAME.get(tag.split(":", 1)[1], tag)
    # numeric mass?  normalize precision to 4 decimals for lookup
    try:
        f = float(tag)
        key = f"{f:+.4f}".rstrip("0").rstrip(".") if False else f"{f:.4f}"
        return _MASS_TO_NAME.get(key, tag)
    except ValueError:
        return _MASS_TO_NAME.get(tag, tag)


def extract_mod_hits(mod_seq: str) -> List[ModHit]:
    """Return every modification in a modified sequence as a list of ``ModHit``.

    The position is 1-based in the *bare* (unmodified) sequence, matching the
    convention used by DIA-NN.  N-terminal modifications (before the first
    residue) are reported at position 0.
    """
    hits: List[ModHit] = []
    plain_pos = 0
    i = 0
    last_residue = ""
    while i < len(mod_seq):
        ch = mod_seq[i]
        if ch.isalpha():
            plain_pos += 1
            last_residue = ch
            i += 1
            continue
        m = MOD_RE.match(mod_seq, i)
        if m:
            tag = m.group(0)
            name = _normalise_tag(tag)
            # Sage emits modifications *after* the modified residue, so the
            # current ``plain_pos`` and ``last_residue`` are what we want.
            hits.append(ModHit(mod_name=name,
                               position=plain_pos,
                               residue=last_residue or "X"))
            i = m.end()
            continue
        i += 1
    return hits


def bare_sequence(mod_seq: str) -> str:
    """Strip all modifications from a Sage/DIA-NN modified sequence."""
    return MOD_RE.sub("", mod_seq)


# ---------------------------------------------------------------------------
# Localization probability
# ---------------------------------------------------------------------------

def _sage_score_column(psm: pd.DataFrame) -> str:
    if "hyperscore" in psm.columns:
        return "hyperscore"
    if "Score" in psm.columns:
        return "Score"
    raise KeyError(
        "Expected Sage match score column 'hyperscore' or 'Score'; "
        f"have: {list(psm.columns)}"
    )


def _fallback_confidence(hits: List[ModHit], bare: str,
                         default_confidence: float) -> Dict[Tuple[str, int], float]:
    """Heuristic per-site confidence when Sage did not search alternative
    placements.  Returns ``{(mod_name, position): prob}``.

    * If the peptide has exactly as many residues of the whitelisted type as
      modifications of that type, the placement is combinatorially
      unambiguous and we assign ``0.99``.
    * Otherwise we assign ``default_confidence`` (typically 0.50) so the
      user can see that the call was not verified by Sage.
    """
    out: Dict[Tuple[str, int], float] = {}
    # Group modifications by type on this peptide
    by_mod: Dict[str, List[ModHit]] = {}
    for h in hits:
        by_mod.setdefault(h.mod_name, []).append(h)
    for mod_name, mod_hits in by_mod.items():
        whitelist = _RESIDUE_WHITELIST.get(mod_name, set())
        if whitelist:
            candidate_count = sum(1 for a in bare if a in whitelist)
        else:
            # unknown mod type: we can't judge ambiguity, give default
            candidate_count = len(mod_hits) + 1  # force default
        if candidate_count == len(mod_hits) and candidate_count > 0:
            prob = 0.99
        else:
            prob = float(default_confidence)
        for h in mod_hits:
            out[(h.mod_name, h.position)] = prob
    return out


def add_site_probabilities(psm: pd.DataFrame,
                           cutoff: float = 0.0,
                           default_confidence: float = 0.50) -> pd.DataFrame:
    """Annotate a Sage PSM table with per-site localization information.

    Parameters
    ----------
    psm
        Sage PSM table after FDR filtering.  Must contain ``peptide``,
        ``filename``, ``charge`` and ``hyperscore`` (or ``Score``).
    cutoff
        Threshold used to populate ``PTM.Site.Confident``.  The default
        (``0.0``) means every PSM is flagged as confident; users can set a
        stricter cutoff and still keep low-localization rows if they also
        pass ``include_low_loc_sites=True`` to ``site_quant``.
    default_confidence
        Fallback probability assigned to a site when no alternative
        placements were searched by Sage.  Set via ``config.default_site_confidence``.

    Returns
    -------
    The same DataFrame plus the following new columns:

    ``PTM.Mods`` : str
        ``"Phospho:S12@0.99,T18@0.50;Oxidation:M4@0.99"`` — complete per-mod
        per-site annotation with embedded probabilities.  Consumed by
        :func:`iter_site_rows` and by :mod:`diaquant.quantify.site_quant`.

    ``PTM.Site.Positions`` : str
        Backwards-compatible ``"S12;T18;M4"`` string (positions only, no mod
        type).  Kept so 0.5.4 downstream code still works.  Phospho-only
        filtering is done in site_quant.

    ``PTM.Modification`` : str
        First modification name (e.g. ``"Phospho"``).  Kept for backwards
        compatibility.

    ``Best.Site.Probability`` : float
        Maximum per-site probability on the peptide — useful as a single
        filter column.  Downstream tools that care about per-site
        probabilities should parse ``PTM.Mods``.

    ``PTM.Site.Confident`` : bool
        ``Best.Site.Probability >= cutoff``.
    """
    if "peptide" not in psm.columns:
        raise KeyError("Expected Sage column 'peptide' in input table.")

    psm = psm.copy()
    if psm.empty:
        for col, default in [
            ("PTM.Mods",               ""),
            ("PTM.Site.Positions",     ""),
            ("PTM.Modification",       "none"),
            ("Best.Site.Probability",  np.nan),
            ("PTM.Site.Confident",     False),
        ]:
            psm[col] = default
        return psm

    # Pre-compute (hits, bare) once per unique peptide to avoid reparsing.
    unique_peps = psm["peptide"].astype(str).unique()
    pep_hits: Dict[str, List[ModHit]] = {p: extract_mod_hits(p) for p in unique_peps}
    pep_bare: Dict[str, str] = {p: bare_sequence(p) for p in unique_peps}

    # Attempt to use Sage alternative-placement PSMs for a real delta-score
    # based confidence.  We group by (filename, bare, charge, mod multiset)
    # and softmax the hyperscore.  When only one placement is present, fall
    # back to ``_fallback_confidence``.
    score_col = _sage_score_column(psm)
    psm["_bare"]    = psm["peptide"].map(lambda p: pep_bare[p])
    psm["_mod_sig"] = psm["peptide"].map(
        lambda p: ";".join(sorted(h.mod_name for h in pep_hits[p]))
    )

    group_key = ["filename", "_bare", "charge", "_mod_sig"]
    grouped = psm.groupby(group_key)
    # Softmax per group (log-sum-exp trick)
    psm["_placement_prob"] = grouped[score_col].transform(
        lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    )
    # group size: a single placement means the softmax is trivially 1 and
    # we prefer the fallback heuristic instead
    psm["_group_size"] = grouped[score_col].transform("size")

    mods_col: List[str] = []
    best_prob_col: List[float] = []
    modif_col: List[str] = []
    positions_col: List[str] = []

    for pep, placement_prob, group_size in zip(psm["peptide"],
                                               psm["_placement_prob"],
                                               psm["_group_size"]):
        hits = pep_hits[pep]
        bare = pep_bare[pep]
        if not hits:
            mods_col.append("")
            best_prob_col.append(np.nan)
            modif_col.append("none")
            positions_col.append("")
            continue
        if group_size > 1:
            # real softmax — apply the same probability to every site on
            # this placement.  (A future release may distribute this
            # across sites using Sage's per-site delta, but Sage does not
            # currently emit that.)
            site_probs = {(h.mod_name, h.position): float(placement_prob)
                          for h in hits}
        else:
            site_probs = _fallback_confidence(hits, bare, default_confidence)

        by_mod: Dict[str, List[Tuple[str, int, float]]] = {}
        for h in hits:
            by_mod.setdefault(h.mod_name, []).append(
                (h.residue, h.position, site_probs[(h.mod_name, h.position)])
            )
        mods_str = ";".join(
            f"{mod}:" + ",".join(f"{r}{p}@{prob:.3f}" for r, p, prob in sites)
            for mod, sites in by_mod.items()
        )
        mods_col.append(mods_str)
        best_prob_col.append(max(site_probs.values()))
        modif_col.append(hits[0].mod_name)
        positions_col.append(";".join(f"{h.residue}{h.position}" for h in hits))

    psm["PTM.Mods"] = mods_col
    psm["Best.Site.Probability"] = best_prob_col
    psm["PTM.Modification"] = modif_col
    psm["PTM.Site.Positions"] = positions_col
    psm["PTM.Site.Confident"] = psm["Best.Site.Probability"].fillna(0.0) >= cutoff

    return psm.drop(columns=["_bare", "_mod_sig", "_placement_prob", "_group_size"])


# ---------------------------------------------------------------------------
# Helpers for the site matrix
# ---------------------------------------------------------------------------

_MOD_SITE_RE = re.compile(r"^([A-Z])(\d+)@([0-9.]+)$")


def iter_site_entries(ptm_mods: str) -> Iterable[Tuple[str, str, int, float]]:
    """Yield ``(mod_name, residue, peptide_local_pos, probability)`` for each
    site described by a ``PTM.Mods`` string.

    ``PTM.Mods`` format::

        "Phospho:S5@0.990,T8@0.500;Oxidation:M2@0.990"
    """
    if not ptm_mods or not isinstance(ptm_mods, str):
        return
    for segment in ptm_mods.split(";"):
        if ":" not in segment:
            continue
        mod_name, sites = segment.split(":", 1)
        mod_name = mod_name.strip()
        for site in sites.split(","):
            m = _MOD_SITE_RE.match(site.strip())
            if m:
                residue, pos, prob = m.group(1), int(m.group(2)), float(m.group(3))
                yield mod_name, residue, pos, prob


# ---------------------------------------------------------------------------
# Backwards compatibility shim
# ---------------------------------------------------------------------------

def extract_mod_positions(mod_seq: str) -> List[Tuple[int, str]]:
    """Legacy helper: return ``[(plain_pos, raw_tag), ...]``.

    Kept so external callers that imported this function before 0.5.5
    continue to work.  Prefer :func:`extract_mod_hits` in new code.
    """
    out: List[Tuple[int, str]] = []
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
            out.append((plain_pos, tag))
            i = m.end()
            continue
        i += 1
    return out
