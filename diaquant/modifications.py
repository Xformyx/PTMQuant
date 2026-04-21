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
}


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
