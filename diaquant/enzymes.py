"""Enzyme catalog for diaquant (NEW in 0.5.1).

Prior to 0.5.1 the pipeline only supported ``trypsin``, ``lys-c`` and
``no-cleavage``, hard-coded inside ``sage_runner.py``.  This module centralises
the full set of proteases commonly used in DIA proteomics so that every
downstream module (``sage_runner``, ``cli``, ``README``) can look them up by a
short, stable identifier.

The catalog closely mirrors the proteases exposed by DIA-NN, MaxQuant and the
upstream Sage engine itself.  Each entry specifies:

* ``cleave_at``
    Sage-style list of one-letter residues (concatenated) at whose
    C-terminus the enzyme cuts.  ``"$"`` is a Sage sentinel meaning "do
    not cut anywhere" (i.e. treat the whole protein as a single peptide,
    useful for top-down and for unspecific searches).
* ``restrict``
    Optional residue that, if present on the *following* position, prevents
    cleavage.  ``"P"`` for Trypsin/LysC encodes the classical "…K|P" and
    "…R|P" exceptions; ``None`` means no restriction.
* ``default_missed_cleavages``
    Sensible per-enzyme default when the user does not override it at
    the global or per-pass level.  This is only applied when the field is
    left untouched in the YAML (``missed_cleavages: null``) to preserve
    backwards compatibility with existing 0.4 / 0.5.0 configurations.

No state is kept here: the catalog is a pure data structure, which keeps
unit tests trivial and lets us grow the list without touching the rest of
the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class EnzymeRule:
    """Minimal description of a protease for Sage."""

    name: str                                        # canonical short id used in YAML / CLI
    cleave_at: str                                   # residues (concatenated) at whose C-terminus the enzyme cuts
    restrict: Optional[str] = None                   # residue that blocks cleavage when it follows
    default_missed_cleavages: int = 2                # reasonable default when the YAML leaves it null
    description: str = ""                            # one-line summary shown by `diaquant list-enzymes`


ENZYME_CATALOG: Dict[str, EnzymeRule] = {
    # ---- canonical tryptic digests ----------------------------------------
    "trypsin": EnzymeRule(
        name="trypsin",
        cleave_at="KR",
        restrict="P",
        default_missed_cleavages=2,
        description="Trypsin/P: cleaves at K/R C-terminus unless followed by P (DIA-NN default).",
    ),
    "trypsin-strict": EnzymeRule(
        name="trypsin-strict",
        cleave_at="KR",
        restrict=None,
        default_missed_cleavages=2,
        description="Trypsin (strict): cleaves at every K/R C-terminus, including K|P and R|P (MaxQuant 'Trypsin').",
    ),
    "lys-c": EnzymeRule(
        name="lys-c",
        cleave_at="K",
        restrict="P",
        default_missed_cleavages=2,
        description="Lys-C/P: cleaves at K C-terminus unless followed by P.",
    ),
    "lys-c-strict": EnzymeRule(
        name="lys-c-strict",
        cleave_at="K",
        restrict=None,
        default_missed_cleavages=2,
        description="Lys-C (strict): cleaves at every K C-terminus.",
    ),
    # ---- less common but commercially relevant ---------------------------
    "arg-c": EnzymeRule(
        name="arg-c",
        cleave_at="R",
        restrict="P",
        default_missed_cleavages=2,
        description="Arg-C: cleaves at R C-terminus unless followed by P.",
    ),
    "asp-n": EnzymeRule(
        name="asp-n",
        cleave_at="D",
        restrict=None,
        default_missed_cleavages=2,
        description="Asp-N: cleaves N-terminally of D. In Sage this is approximated as cleavage at D C-terminus; use with care.",
    ),
    "glu-c": EnzymeRule(
        name="glu-c",
        cleave_at="E",
        restrict=None,
        default_missed_cleavages=2,
        description="Glu-C (V8): cleaves at E C-terminus (in bicarbonate buffer).",
    ),
    "chymotrypsin": EnzymeRule(
        name="chymotrypsin",
        cleave_at="FWYL",
        restrict="P",
        default_missed_cleavages=2,
        description="Chymotrypsin: cleaves at F/W/Y/L C-terminus unless followed by P.",
    ),
    # ---- unspecific / top-down ------------------------------------------
    "no-cleavage": EnzymeRule(
        name="no-cleavage",
        cleave_at="$",     # Sage sentinel: no enzymatic cleavage
        restrict=None,
        default_missed_cleavages=0,
        description="Unspecific / no enzymatic cleavage (top-down, open search).",
    ),
}


def get_enzyme(name: str) -> EnzymeRule:
    """Return the :class:`EnzymeRule` for ``name`` or raise with a helpful list."""
    key = name.strip().lower()
    if key not in ENZYME_CATALOG:
        known = ", ".join(sorted(ENZYME_CATALOG))
        raise ValueError(
            f"Unknown enzyme '{name}'. Valid choices: {known}."
        )
    return ENZYME_CATALOG[key]


def list_enzymes() -> Dict[str, EnzymeRule]:
    """Return the entire catalog (used by ``diaquant list-enzymes``)."""
    return dict(ENZYME_CATALOG)


def to_sage_enzyme_block(
    rule: EnzymeRule,
    *,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
) -> Dict[str, object]:
    """Translate a :class:`EnzymeRule` into the JSON block Sage v0.14 expects."""
    return {
        "missed_cleavages": missed_cleavages,
        "min_len": min_len,
        "max_len": max_len,
        "cleave_at": rule.cleave_at,
        "restrict": rule.restrict,
    }
