"""Search-parameter profiles per PTM family.

Each PTM has a different optimal search configuration.  For example,
ubiquitin di-glycyl (K-GG) almost always requires ``missed_cleavages = 3``
because the modification blocks tryptic cleavage at the modified lysine,
while phosphorylation needs an enriched peptide-length range and stricter
site-localisation.  Hard-coding a single Sage configuration would either be
too sensitive (false positives) or miss real PTM hits.

This module ships a small registry of *passes*: each pass declares the set
of variable modifications to enable plus pass-specific overrides for things
like missed cleavages, max variable modifications, and peptide-length
limits.  Users select which passes to run from their YAML config; only
those passes execute, and their results are merged into a single set of
DIA-NN-style output tables at the end of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PassProfile:
    """One PTM-search pass.

    All fields except ``name`` and ``variable_modifications`` fall back to the
    user's global :class:`~diaquant.config.DiaQuantConfig` defaults when set
    to ``None``.
    """

    name: str                                                # e.g. "phospho"
    variable_modifications: List[str]                        # built-in mod names
    description: str = ""
    missed_cleavages: Optional[int] = None
    max_variable_mods: Optional[int] = None
    min_peptide_length: Optional[int] = None
    max_peptide_length: Optional[int] = None
    max_precursor_charge: Optional[int] = None
    site_probability_cutoff: Optional[float] = None
    fragment_tol_ppm: Optional[float] = None
    # Whether this pass is the "whole-proteome" backbone used for
    # protein-group quantification.  Only one pass should set this True.
    is_whole_proteome: bool = False


# ---------------------------------------------------------------------------
# Built-in pass catalogue -- tuned for whole-proteome DIA on Thermo Orbitrap
# Exploris 240 with 6 m/z windows.  Defaults follow Bekker-Jensen 2020 (MCP)
# and the latest AlphaPeptDeep / DIA-NN community recommendations.
# ---------------------------------------------------------------------------
PASS_PROFILES: Dict[str, PassProfile] = {
    "whole_proteome": PassProfile(
        name="whole_proteome",
        description=(
            "Backbone pass for protein-level quantification.  Only "
            "M oxidation and protein N-terminal acetylation are enabled "
            "as variable mods so that the search space stays small and "
            "the resulting precursor table can be used directly for "
            "directLFQ protein roll-up."
        ),
        variable_modifications=["Oxidation", "Acetyl_Nterm"],
        missed_cleavages=2,
        max_variable_mods=2,
        min_peptide_length=7,
        max_peptide_length=30,
        max_precursor_charge=4,
        is_whole_proteome=True,
    ),
    "phospho": PassProfile(
        name="phospho",
        description=(
            "STY phosphorylation pass.  Allows up to 3 variable mods and a "
            "stricter site probability cutoff (0.75) consistent with the "
            "Bekker-Jensen 2020 phospho-DIA workflow."
        ),
        variable_modifications=["Oxidation", "Acetyl_Nterm", "Phospho"],
        missed_cleavages=2,
        max_variable_mods=3,
        min_peptide_length=7,
        max_peptide_length=30,
        max_precursor_charge=4,
        site_probability_cutoff=0.75,
    ),
    "ubiquitin": PassProfile(
        name="ubiquitin",
        description=(
            "K-GlyGly (UniMod:121) ubiquitin remnant pass.  Missed "
            "cleavages raised to 3 because the GG tag blocks tryptic "
            "cleavage at the modified lysine."
        ),
        variable_modifications=["Oxidation", "Acetyl_Nterm", "GlyGly"],
        missed_cleavages=3,
        max_variable_mods=3,
        min_peptide_length=7,
        max_peptide_length=35,
        max_precursor_charge=4,
        site_probability_cutoff=0.75,
    ),
    "acetyl_methyl": PassProfile(
        name="acetyl_methyl",
        description=(
            "Lysine acetylation plus K/R mono/di/tri-methylation.  These "
            "PTMs share the property of blocking tryptic cleavage at the "
            "modified lysine, so missed cleavages are raised to 3.  This "
            "is the headline pass that DIA-NN cannot quantify reliably."
        ),
        variable_modifications=[
            "Oxidation", "Acetyl_Nterm",
            "Acetyl", "Methyl", "Dimethyl", "Trimethyl",
        ],
        missed_cleavages=3,
        max_variable_mods=3,
        min_peptide_length=7,
        max_peptide_length=35,
        max_precursor_charge=4,
        site_probability_cutoff=0.75,
    ),
    "succinyl_acyl": PassProfile(
        name="succinyl_acyl",
        description=(
            "K-acyl pass: succinyl, malonyl, crotonyl.  All block tryptic "
            "cleavage at K, so missed cleavages set to 3."
        ),
        variable_modifications=[
            "Oxidation", "Acetyl_Nterm",
            "Succinyl", "Malonyl", "Crotonyl",
        ],
        missed_cleavages=3,
        max_variable_mods=2,
        min_peptide_length=7,
        max_peptide_length=35,
        max_precursor_charge=4,
        site_probability_cutoff=0.75,
    ),
}


def list_builtin_passes() -> List[str]:
    """Return the names of all registered built-in passes."""
    return list(PASS_PROFILES.keys())


def resolve_passes(selected: List[str],
                   custom: List[dict] = ()) -> List[PassProfile]:
    """Resolve the user's pass selection from the YAML config.

    ``selected`` is the list of built-in pass names; ``custom`` is the list
    of fully-specified pass dictionaries from ``custom_passes:`` in YAML.
    Unknown names raise ``KeyError`` with a helpful message.
    """
    out: List[PassProfile] = []
    for name in selected or []:
        if name not in PASS_PROFILES:
            raise KeyError(
                f"Unknown pass '{name}'. "
                f"Pick from {list_builtin_passes()} or define it under "
                f"'custom_passes:' in the YAML."
            )
        out.append(PASS_PROFILES[name])
    for item in custom or []:
        out.append(
            PassProfile(
                name=item["name"],
                variable_modifications=list(item["variable_modifications"]),
                description=item.get("description", ""),
                missed_cleavages=item.get("missed_cleavages"),
                max_variable_mods=item.get("max_variable_mods"),
                min_peptide_length=item.get("min_peptide_length"),
                max_peptide_length=item.get("max_peptide_length"),
                max_precursor_charge=item.get("max_precursor_charge"),
                site_probability_cutoff=item.get("site_probability_cutoff"),
                fragment_tol_ppm=item.get("fragment_tol_ppm"),
                is_whole_proteome=bool(item.get("is_whole_proteome", False)),
            )
        )
    if not out:
        raise ValueError(
            "No pass selected. Add at least one entry under 'passes:' "
            "(e.g. 'whole_proteome') in your YAML configuration."
        )
    return out
