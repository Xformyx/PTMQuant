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
from .predicted_library import fine_tune_models, generate_predicted_library
from .ptm_profiles import PassProfile, resolve_passes
from .rescore import rescore_with_predicted_library
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
    # v0.5.8: per-pass peptide FDR (None inherits the global default).
    if getattr(profile, "peptide_fdr", None) is not None:
        cfg.peptide_fdr = profile.peptide_fdr
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
    # v0.5.5: also keep the pre-FDR PSM table per pass, for MBR donor pooling.
    per_pass_full: Dict[str, pd.DataFrame] = {}
    # v0.5.4: remember every predicted-library TSV we generated/reused so the
    # caller (cli.run) can drop copies into out_dir for user-side verification.
    library_paths: List[Path] = []
    # Diagnostic counters the caller writes into run_manifest.json
    n_psms_raw_total = 0
    n_psms_rescored_total = 0
    for i, profile in enumerate(profiles):
        print(f"[diaquant] -> pass '{profile.name}': "
              f"vars={profile.variable_modifications} "
              f"missed_cleavages={profile.missed_cleavages or base.missed_cleavages}")
        pass_cfg = _config_for_pass(base, profile)

        # ---- 0.5.0: generate predicted spectral library before searching --
        # The TSV is written inside the pass's output dir and also reused
        # after the Sage search for post-hoc rescoring.  A ``None`` return
        # simply means we fall back to the 0.4.x behaviour (no rescoring).
        library_tsv = None
        if pass_cfg.predicted_library:
            try:
                library_tsv = generate_predicted_library(
                    pass_cfg,
                    out_dir=pass_cfg.output_dir,
                    pass_label=profile.name,
                )
            except Exception as exc:                                # pragma: no cover
                if pass_cfg.pred_lib_fallback_in_silico:
                    print(f"[diaquant]    predicted library failed ({exc}); "
                          f"falling back to Sage built-in.")
                    library_tsv = None
                else:
                    raise
            if library_tsv is not None:
                library_paths.append(Path(library_tsv))

        # Resume mode: reuse existing Sage results if present; otherwise
        # run the (auto-batched) Sage search so both the 0.4.x resume
        # feature and the 0.5.0 predicted-library feature stay compatible.
        cached_tsv = pass_cfg.output_dir / "sage" / "results.sage.tsv"
        if resume and cached_tsv.exists():
            print(f"[diaquant] -> pass '{profile.name}': "
                  f"cached results found ({cached_tsv}), skipping Sage search.")
            sage_tsv = cached_tsv
        else:
            sage_tsv = run_sage_batched(pass_cfg)

        df, df_full = parse_sage_tsv(sage_tsv,
                                     site_cutoff=pass_cfg.site_probability_cutoff,
                                     peptide_fdr=pass_cfg.peptide_fdr,
                                     return_unfiltered=True)
        df = attach_fasta_meta(df, pass_cfg.fasta)
        df_full = _annotate_pass(df_full, profile)
        per_pass_full[profile.name] = df_full
        n_psms_raw_total += len(df)

        # ---- 0.5.0: post-hoc rescoring with AlphaPeptDeep RT prediction ----
        if pass_cfg.rescore_with_prediction and library_tsv is not None:
            df = rescore_with_predicted_library(df, library_tsv, pass_cfg)
            n_psms_rescored_total += len(df)

        df = _annotate_pass(df, profile)
        per_pass[profile.name] = df
        print(f"[diaquant]    pass '{profile.name}': "
              f"{len(df)} precursor rows after FDR.")

        # ---- 0.5.0: optional transfer learning after the first pass -------
        # Fine-tunes RT / MS2 on the user's own high-confidence PSMs so that
        # every *subsequent* pass re-predicts with run-adapted weights.
        if (i == 0
                and pass_cfg.pred_lib_transfer_learning
                and not df.empty):
            hi_conf = df[df.get("Q.Value",
                                pd.Series([0.0]*len(df))).astype(float) <= 0.01]
            fine_tune_models(pass_cfg, hi_conf)

    if not per_pass:
        raise RuntimeError("No passes produced output.")

    merged = pd.concat(per_pass.values(), ignore_index=True, sort=False)
    # pandas.concat drops ``.attrs``; reattach the FASTA records from the first
    # pass that carries them so site_quant can compute absolute PTM positions.
    for _pass_df in per_pass.values():
        rec = getattr(_pass_df, "attrs", {}).get("fasta_records")
        if rec:
            merged.attrs["fasta_records"] = rec
            break

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
    # v0.5.4: stash diagnostics on the merged frame so cli.run can surface them
    # in run_manifest.json without plumbing more return values through.
    merged.attrs["predicted_library_paths"] = library_paths
    merged.attrs["n_psms_raw_total"] = n_psms_raw_total
    merged.attrs["n_psms_rescored_total"] = n_psms_rescored_total
    if per_pass_full:
        merged.attrs["psm_full"] = pd.concat(
            per_pass_full.values(), ignore_index=True, sort=False,
        )
    return merged, per_pass
