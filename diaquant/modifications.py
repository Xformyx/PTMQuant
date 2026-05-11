"""Built-in and user-defined post-translational modification (PTM) definitions.

Each modification is described by a small dataclass that contains everything Sage
needs to register the variable / fixed modification, plus the human-readable
identifier (UniMod accession when available) that we use throughout the
DIA-NN-compatible output (e.g. the ``Modified.Sequence`` column).

The default catalogue covers the PTMs that DIA-NN handles natively
(phosphorylation, ubiquitylation/GlyGly) plus the modifications that DIA-NN is
known to be weak on (acetylation, mono/di/tri methylation, succinylation,
malonylation, crotonylation, SUMO/GG remnants, etc.).  The user can extend the
catalogue from a YAML configuration file by simply listing the UniMod ID,
mass shift, target residues and whether the modification should be treated as
fixed or variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Modification:
    """A mass-spec modification descriptor."""

    name: str                     # human readable, e.g. "Phospho"
    unimod_id: Optional[int]      # UniMod accession (None for custom)
    mass_shift: float             # monoisotopic Δmass in Da
    targets: tuple                # residues, e.g. ("S","T","Y") or ("K",) or ("[",) for N-term
    fixed: bool = False           # True -> fixed mod, False -> variable
    neutral_loss: float = 0.0     # optional neutral loss (Da) for diagnostic ions

    @property
    def display(self) -> str:
        """String used inside Modified.Sequence: e.g. ``S(UniMod:21)``."""
        return f"UniMod:{self.unimod_id}" if self.unimod_id else self.name


# ---- Default catalogue -------------------------------------------------------
# Mass values come from the public UniMod database (https://www.unimod.org).
DEFAULT_MODIFICATIONS: Dict[str, Modification] = {
    # fixed
    "Carbamidomethyl":  Modification("Carbamidomethyl", 4,  57.021464, ("C",), fixed=True),
    # variable, well-supported
    "Oxidation":        Modification("Oxidation",       35, 15.994915, ("M",)),
    "Acetyl_Nterm":     Modification("Acetyl_Nterm",     1, 42.010565, ("[",)),
    "Phospho":          Modification("Phospho",         21, 79.966331, ("S", "T", "Y")),
    "GlyGly":           Modification("GlyGly",         121, 114.042927, ("K",)),
    # PTMs DIA-NN is weak on
    "Acetyl":           Modification("Acetyl",          1, 42.010565, ("K",)),
    "Methyl":           Modification("Methyl",         34, 14.015650, ("K", "R")),
    "Dimethyl":         Modification("Dimethyl",       36, 28.031300, ("K", "R")),
    "Trimethyl":        Modification("Trimethyl",      37, 42.046950, ("K",)),
    "Succinyl":         Modification("Succinyl",       64, 100.016044, ("K",)),
    "Malonyl":          Modification("Malonyl",        747, 86.000394, ("K",)),
    "Crotonyl":         Modification("Crotonyl",       1363, 68.026215, ("K",)),
    "Sumo_QQTGG":       Modification("Sumo_QQTGG",    1340, 471.207606, ("K",)),
    "Citrullination":   Modification("Citrullination",  7, 0.984016, ("R",)),
    # extra built-ins added in 0.5.0 so the corresponding passes work out-of-the-box
    "OGlcNAc":          Modification("OGlcNAc",        43, 203.079373, ("S", "T")),
    "Lactyl":           Modification("Lactyl",       2114, 72.021129, ("K",)),
    "Propionyl":        Modification("Propionyl",      58, 56.026215, ("K",)),
    "Butyryl":          Modification("Butyryl",     1289, 70.041865, ("K",)),
    "Sulfation":        Modification("Sulfation",      40, 79.956815, ("Y",)),
}


# ---- AlphaDIA -> DIA-NN modified-sequence formatter (v0.6.0 Phase 3) ----
# Reverse lookup: modification name -> UniMod ID, built once from the catalogue.
_NAME_TO_UNIMOD: Dict[str, int] = {
    name: mod.unimod_id
    for name, mod in DEFAULT_MODIFICATIONS.items()
    if mod.unimod_id is not None
}
# Some AlphaDIA / alphabase mod names map onto multiple PTMQuant entries
# (e.g. ``Acetyl`` is used both for K side-chain and protein N-term).  We
# resolve them at format time using the residue / position context.
_ALPHA_NAME_ALIASES = {
    "Acetyl@Protein_N-term": 1,   # UniMod:1 (Acetyl on protein N-terminus)
    "Acetyl@Protein N-term": 1,
    "Acetyl@Any_N-term":     1,
    "Acetyl@Any N-term":     1,
    "Carbamidomethyl@C":     4,   # fixed mod -- usually omitted from PSM mods
    "Oxidation@M":          35,
    "Phospho@S":            21, "Phospho@T": 21, "Phospho@Y": 21,
    "GlyGly@K":            121,
    "Acetyl@K":              1,   # K side-chain acetyl shares UniMod:1
    "Methyl@K":             34, "Methyl@R": 34,
    "Dimethyl@K":           36, "Dimethyl@R": 36,
    "Trimethyl@K":          37,
}


def _resolve_unimod(mod_name: str) -> Optional[int]:
    """Return the UniMod accession for an AlphaDIA mod token.

    AlphaDIA emits mods either as a bare name (``"Phospho"``) or as a fully
    qualified ``"Name@Target"`` token (``"Acetyl@Protein_N-term"``).  We try the
    qualified alias first so we can disambiguate Acetyl-N-term (UniMod:1) from
    K-side-chain Acetyl (also UniMod:1, but documented separately upstream).
    Returns ``None`` for unknown mods so the caller can fall back to the bare
    name -- never a silent KeyError.
    """
    if not mod_name:
        return None
    if mod_name in _ALPHA_NAME_ALIASES:
        return _ALPHA_NAME_ALIASES[mod_name]
    bare = mod_name.split("@", 1)[0]
    return _NAME_TO_UNIMOD.get(bare)


def format_diann_sequence(sequence: str,
                          mods: str,
                          mod_sites: str) -> str:
    """Format an AlphaDIA ``(sequence, mods, mod_sites)`` triple into the
    DIA-NN-style ``Modified.Sequence`` string.

    The DIA-NN convention used throughout PTMQuant is:

    * Bare sequence wrapped in underscores (``_PEPTIDE_``).
    * Modifications inline as ``(UniMod:N)`` immediately *after* the residue
      they sit on; e.g. ``_PEPS(UniMod:21)IDE_`` for Phospho-on-S at site 4.
    * N-terminal modifications appear at the very start, *before* the first
      residue: ``_(UniMod:1)PEPTIDE_``.  AlphaDIA encodes the N-term position
      as ``mod_site == "0"``.
    * C-terminal modifications appear at the very end, *after* the last
      residue: ``_PEPTIDE(UniMod:30)_``.  AlphaDIA encodes the C-term as
      ``mod_site == "-1"`` (or ``len(sequence) + 1`` in some library variants).

    Unknown modifications fall back to ``(ModName)`` so nothing is silently
    dropped -- the resulting peptide is still a valid DIA-NN modified-sequence
    token (DIA-NN itself accepts custom names enclosed in parens).

    Parameters
    ----------
    sequence
        Bare amino-acid sequence (e.g. ``"AASPEPTIDER"``).
    mods
        Semicolon-separated mod names from AlphaDIA's ``precursor.mods``
        column.  Empty string / NaN means no modifications.
    mod_sites
        Semicolon-separated 1-based residue positions matching ``mods``.
        Special values: ``"0"`` for protein N-term, ``"-1"`` for C-term.
    """
    if not isinstance(sequence, str) or not sequence:
        return ""

    # No mods -> just wrap in underscores so the column type stays consistent.
    if not isinstance(mods, str) or not mods.strip():
        return f"_{sequence}_"
    if not isinstance(mod_sites, str) or not mod_sites.strip():
        return f"_{sequence}_"

    mod_list = [m for m in mods.split(";") if m]
    site_list = [s for s in mod_sites.split(";") if s != ""]
    if len(mod_list) != len(site_list):
        # Schema violation -- play it safe and return the bare sequence.
        return f"_{sequence}_"

    nterm_tags: List[str] = []
    cterm_tags: List[str] = []
    # site -> list of tags (peptides can carry multiple mods per residue,
    # though it's vanishingly rare in PTM passes -- preserve order).
    inline_tags: Dict[int, List[str]] = {}

    for mod_name, site_str in zip(mod_list, site_list):
        unimod = _resolve_unimod(mod_name)
        bare_name = mod_name.split("@", 1)[0]
        tag = f"(UniMod:{unimod})" if unimod is not None else f"({bare_name})"
        try:
            site = int(site_str)
        except (TypeError, ValueError):
            continue
        if site == 0:
            nterm_tags.append(tag)
        elif site == -1 or site > len(sequence):
            cterm_tags.append(tag)
        else:
            inline_tags.setdefault(site, []).append(tag)

    # Build the formatted string.  We walk left-to-right so the inline tags
    # land in the right slot.
    parts: List[str] = ["_"]
    parts.extend(nterm_tags)
    for idx, residue in enumerate(sequence, start=1):
        parts.append(residue)
        if idx in inline_tags:
            parts.extend(inline_tags[idx])
    parts.extend(cterm_tags)
    parts.append("_")
    return "".join(parts)


def parse_user_modifications(items: Iterable[dict]) -> List[Modification]:
    """Parse a list of dictionaries from the YAML config into Modification objects.

    Each dict accepts the keys: name, unimod_id, mass_shift, targets, fixed,
    neutral_loss.  ``targets`` may be a list (e.g. ``["S","T","Y"]``) or a
    single string.  Any field not provided falls back to a sensible default.
    """
    mods: List[Modification] = []
    for it in items or []:
        name = it["name"]
        targets = it.get("targets", [])
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        mods.append(
            Modification(
                name=name,
                unimod_id=it.get("unimod_id"),
                mass_shift=float(it["mass_shift"]),
                targets=tuple(targets),
                fixed=bool(it.get("fixed", False)),
                neutral_loss=float(it.get("neutral_loss", 0.0)),
            )
        )
    return mods


def resolve_modifications(active_names: Iterable[str],
                          extra: Iterable[dict] = ()) -> List[Modification]:
    """Combine built-in mods (selected by name) with user-defined ones.

    Unknown names raise ``KeyError`` so the user gets immediate feedback.
    """
    selected: List[Modification] = []
    for name in active_names or []:
        if name not in DEFAULT_MODIFICATIONS:
            raise KeyError(
                f"Unknown built-in modification '{name}'. "
                f"Add it to 'custom_modifications:' in the YAML or pick from "
                f"{list(DEFAULT_MODIFICATIONS)}."
            )
        selected.append(DEFAULT_MODIFICATIONS[name])
    selected.extend(parse_user_modifications(extra))
    return selected
