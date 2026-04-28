"""Orchestrate multiple PTM-search passes and merge their results.

This is the heart of the diaquant 0.3 architecture.  Instead of dialling
every variable modification on at once (which causes a combinatorial
explosion in the Sage index and degrades sensitivity for *every* PTM), we
run one Sage search per *pass* (e.g. ``whole_proteome``, ``phospho``,
``acetyl_methyl``).  Each pass uses parameters tuned for the PTM family it
targets: phospho needs site-localisation, K-acyl mods need
``missed_cleavages = 3`` because the modification blocks tryptic cleavage,
and so on.  All passes share the same FASTA and the same set of mzML
files.

After every pass finishes, we concatenate the long precursor tables,
de-duplicate by ``Precursor.Id`` (keeping the row with the higher Sage
score), then run directLFQ once for protein roll-up and once per modified
PTM family for site roll-up.  This guarantees that the protein matrix is
identical to a stand-alone whole-proteome search, while every PTM gets its
own optimally-localised site quantification.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import DiaQuantConfig
from .parse_sage import attach_fasta_meta, parse_sage_tsv
from .ptm_profiles import PassProfile, resolve_passes
from .sage_runner import run_sage, run_sage_batched


def _config_for_pass(base: DiaQuantConfig, profile: PassProfile) -> DiaQuantConfig:
    """Return a shallow copy of ``base`` with the pass profile applied."""
    cfg = replace(base)
    cfg.variable_modifications = list(profile.variable_modifications)
    if profile.missed_cleavages is not None:
        cfg.missed_cleavages = profile.missed_cleavages
    if profile.max_variable_mods is not None:
        cfg.max_variable_mods = profile.max_variable_mods
    if profile.min_peptide_length is not None:
        cfg.min_peptide_length = profile.min_peptide_length
    if profile.max_peptide_length is not None:
        cfg.max_peptide_length = profile.max_peptide_length
    if profile.max_precursor_charge is not None:
        cfg.max_precursor_charge = profile.max_precursor_charge
    if profile.site_probability_cutoff is not None:
        cfg.site_probability_cutoff = profile.site_probability_cutoff
    if profile.fragment_tol_ppm is not None:
        cfg.fragment_tol_ppm = profile.fragment_tol_ppm
    # Each pass writes Sage results into its own subdirectory so the
    # different runs do not overwrite each other.
    cfg.output_dir = base.output_dir / f"pass_{profile.name}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _annotate_pass(df: pd.DataFrame, profile: PassProfile) -> pd.DataFrame:
    """Add a ``Pass`` column so we can trace each precursor back to its origin."""
    if df.empty:
        df = df.copy()
        df["Pass"] = pd.Series(dtype="object")
        df["Is.Whole.Proteome.Pass"] = pd.Series(dtype="bool")
        return df
    df = df.copy()
    df["Pass"] = profile.name
    df["Is.Whole.Proteome.Pass"] = bool(profile.is_whole_proteome)
    return df


def run_multipass(base: DiaQuantConfig, resume: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute every selected pass and return the merged long table.

    Parameters
    ----------
    base : DiaQuantConfig
    resume : bool
        When True, skip the Sage search for any pass whose
        ``pass_{name}/sage/results.sage.tsv`` already exists on disk.
        Useful for resuming an interrupted run without re-doing expensive
        database searches.

    Returns
    -------
    merged_long : pandas.DataFrame
        All passes concatenated, de-duplicated on (filename, Precursor.Id).
    per_pass : Dict[str, pandas.DataFrame]
        Per-pass long table, useful for debugging / per-PTM exports.
    """
    profiles = resolve_passes(base.passes, base.custom_passes)
    print(
        f"[diaquant] multi-pass workflow with "
        f"{len(profiles)} pass(es): {[p.name for p in profiles]}"
    )

    per_pass: Dict[str, pd.DataFrame] = {}
    for profile in profiles:
        print(f"[diaquant] -> pass '{profile.name}': "
              f"vars={profile.variable_modifications} "
              f"missed_cleavages={profile.missed_cleavages or base.missed_cleavages}")
        pass_cfg = _config_for_pass(base, profile)

        # Resume mode: reuse existing Sage results if present
        cached_tsv = pass_cfg.output_dir / "sage" / "results.sage.tsv"
        if resume and cached_tsv.exists():
            print(f"[diaquant] -> pass '{profile.name}': "
                  f"cached results found ({cached_tsv}), skipping Sage search.")
            sage_tsv = cached_tsv
        else:
            sage_tsv = run_sage_batched(pass_cfg)

        df = parse_sage_tsv(sage_tsv,
                            site_cutoff=pass_cfg.site_probability_cutoff,
                            peptide_fdr=pass_cfg.peptide_fdr)
        df = attach_fasta_meta(df, pass_cfg.fasta)
        df = _annotate_pass(df, profile)
        per_pass[profile.name] = df
        print(f"[diaquant]    pass '{profile.name}': "
              f"{len(df)} precursor rows after FDR.")

    if not per_pass:
        raise RuntimeError("No passes produced output.")

    merged = pd.concat(per_pass.values(), ignore_index=True, sort=False)

    # When the same (filename, Precursor.Id) appears in multiple passes
    # (typical for unmodified peptides that are reported by every pass),
    # keep the row from the whole-proteome pass when available; otherwise
    # keep the row with the highest Sage discriminant score.
    if "sage_discriminant_score" in merged.columns:
        merged = merged.sort_values(
            ["Is.Whole.Proteome.Pass", "sage_discriminant_score"],
            ascending=[False, False],
        )
    else:
        merged = merged.sort_values(["Is.Whole.Proteome.Pass"], ascending=False)
    merged = merged.drop_duplicates(["filename", "Precursor.Id"], keep="first")
    print(f"[diaquant] merged unique precursors: {len(merged)}")
    return merged, per_pass
