"""diaquant command-line interface.

Two top-level commands:

* ``diaquant init-config`` -- generate a starter YAML.
* ``diaquant run``         -- execute the pipeline.

If the YAML lists one or more entries under ``passes:`` (or
``custom_passes:``) diaquant will automatically run a *multi-pass* PTM
search: one Sage search per pass with PTM-specific parameter overrides,
followed by a single merged directLFQ quantification step.  When no
passes are listed diaquant falls back to the original single-pass
behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import click
import pandas as pd
import yaml

from . import __version__
from .config import DiaQuantConfig
from .modifications import DEFAULT_MODIFICATIONS
from .multipass import run_multipass
from .parse_sage import attach_fasta_meta, parse_sage_tsv
from .ptm_profiles import PASS_PROFILES, list_builtin_passes
from .quantify import precursor_matrix, protein_quant, site_quant
from .rt_align import RTAlignParams, align_runs, write_rt_stats
from .sage_runner import run_sage, run_sage_batched
from .stats import differential, load_sample_sheet
from .writer import (
    PG_COLS,
    PR_COLS,
    write_main_report,
    write_pg_matrix,
    write_pr_matrix,
    write_site_matrix,
)


@click.group()
@click.version_option(__version__, prog_name="diaquant")
def cli() -> None:
    """Open-source DIA proteomics quantification with universal PTM support."""


@cli.command("init-config")
@click.option("--output", "out", required=True, type=click.Path(dir_okay=False))
@click.option("--fasta", required=True, type=click.Path(exists=True))
@click.option("--mzml-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--pass", "passes", multiple=True,
              help=f"Pass(es) to enable. Choose from {list_builtin_passes()}. "
                   f"Repeat the flag for multiple passes.")
@click.option("--ptm", "ptms", multiple=True,
              help="(Single-pass mode) variable PTM(s) to enable. "
                   f"Choose from {list(DEFAULT_MODIFICATIONS)}.")
@click.option("--sample-sheet", default=None, type=click.Path(),
              help="Optional TSV with mzml_file & group columns for "
                   "automatic differential analysis.")
def init_config(out: str, fasta: str, mzml_dir: str,
                passes: List[str], ptms: List[str],
                sample_sheet: str) -> None:
    """Generate a starter YAML configuration."""
    files = sorted(str(p) for p in Path(mzml_dir).glob("*.mzML"))
    if not files:
        raise click.ClickException(f"No .mzML files found in {mzml_dir}")

    cfg = {
        "fasta": fasta,
        "mzml_files": files,
        "output_dir": "diaquant_results",
        "fixed_modifications": ["Carbamidomethyl"],
        "max_variable_mods": 2,
        "missed_cleavages": 2,
        "min_peptide_length": 7,
        "max_peptide_length": 30,
        "min_precursor_charge": 2,
        "max_precursor_charge": 4,
        "min_precursor_mz": 400.0,
        "max_precursor_mz": 1000.0,
        "precursor_tol_ppm": 6.0,
        "fragment_tol_ppm": 12.0,
        "psm_fdr": 0.01,
        "site_probability_cutoff": 0.75,
        "match_between_runs": True,
        "scoring_mode": "peptidoforms",
        "machine_learning": "nn_cv",
        "rt_alignment": True,
        "rt_align_frac": 0.2,
        "rt_align_min_anchors": 50,
        "rt_align_q_cutoff": 0.01,
        "threads": 0,
    }
    if passes:
        cfg["passes"] = list(passes)
        cfg["variable_modifications"] = []
    else:
        cfg["passes"] = []
        cfg["variable_modifications"] = list(ptms) or ["Oxidation", "Acetyl_Nterm"]
    cfg["custom_modifications"] = []
    cfg["custom_passes"] = []
    if sample_sheet:
        cfg["sample_sheet"] = sample_sheet

    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    click.echo(f"Wrote starter config -> {out}")
    if passes:
        click.echo(f"  Passes enabled: {list(passes)}")
    else:
        click.echo(f"  Single-pass variable mods: {cfg['variable_modifications']}")


@cli.command("list-passes")
def list_passes() -> None:
    """List built-in PTM-search passes and their description."""
    for name, profile in PASS_PROFILES.items():
        click.secho(f"- {name}", fg="green", bold=True)
        click.echo(f"    {profile.description}")
        click.echo(f"    variable_modifications = {profile.variable_modifications}")
        if profile.missed_cleavages is not None:
            click.echo(f"    missed_cleavages       = {profile.missed_cleavages}")
        if profile.max_variable_mods is not None:
            click.echo(f"    max_variable_mods      = {profile.max_variable_mods}")


@cli.command("run")
@click.option("--config", "cfg_path", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--resume", is_flag=True, default=False,
              help="Skip Sage searches for passes where results.sage.tsv already exists. "
                   "Useful for resuming an interrupted run.")
def run(cfg_path: str, resume: bool) -> None:
    """Execute the full Sage → directLFQ → DIA-NN-style export pipeline."""
    cfg = DiaQuantConfig.from_yaml(cfg_path)
    click.secho(f"[diaquant {__version__}] starting", fg="green")
    click.echo(f"  fasta : {cfg.fasta}")
    click.echo(f"  mzml  : {len(cfg.mzml_files)} files")
    click.echo(f"  out   : {cfg.output_dir}")
    if resume:
        click.echo(f"  mode  : resume (cached Sage results will be reused)")

    # Auto-batch report
    if cfg.batch_size > 0 and cfg.batch_size < len(cfg.mzml_files):
        import math
        n_batches = math.ceil(len(cfg.mzml_files) / cfg.batch_size)
        click.secho(
            f"  auto-batch: {len(cfg.mzml_files)} files → {n_batches} batches "
            f"of ≤{cfg.batch_size} files (memory-safe mode)",
            fg="cyan",
        )

    if cfg.passes or cfg.custom_passes:
        click.echo(f"  mode  : multi-pass ({cfg.passes + [p['name'] for p in cfg.custom_passes]})")
        long_df, _per_pass = run_multipass(cfg, resume=resume)
    else:
        click.echo(f"  mode  : single-pass (vars={cfg.variable_modifications})")
        cached_tsv = cfg.output_dir / "sage" / "results.sage.tsv"
        if resume and cached_tsv.exists():
            click.echo(f"  resume: cached Sage results found, skipping search.")
            sage_tsv = cached_tsv
        else:
            sage_tsv = run_sage_batched(cfg)
        long_df = parse_sage_tsv(
            sage_tsv,
            site_cutoff=cfg.site_probability_cutoff,
            peptide_fdr=cfg.peptide_fdr,
        )
        long_df = attach_fasta_meta(long_df, cfg.fasta)

    if long_df.empty:
        # Sage found PSMs but all were removed by FDR / other filters.
        # Print a diagnostic summary of what Sage actually found.
        n_files = len(cfg.mzml_files)
        lfq_enabled = n_files >= 2
        click.secho(
            "[diaquant] No target PSMs passed filters (empty result).\n"
            f"  Files searched       : {n_files}\n"
            f"  Sage LFQ mode        : {'enabled (lfq=true)' if lfq_enabled else 'disabled (lfq=false, single file)'}\n"
            f"  Peptide FDR cutoff   : {cfg.peptide_fdr}\n"
            "  Hints:\n"
            "    • Check that ≥2 mzML files are listed when lfq is needed.\n"
            "    • Relax peptide_fdr (e.g. 0.10) and check Sage's results.sage.tsv.\n"
            "    • Confirm mzML is DIA (not DDA) and FASTA matches the species.",
            fg="yellow",
        )
        out = cfg.output_dir
        sample_names = [Path(p).name for p in cfg.mzml_files]
        write_pr_matrix(
            pd.DataFrame(columns=PR_COLS + sample_names),
            out / "report.pr_matrix.tsv",
        )
        pd.DataFrame(columns=PG_COLS + sample_names).to_csv(
            out / "report.pg_matrix.tsv", sep="\t", index=False, na_rep=""
        )
        write_site_matrix(pd.DataFrame(), out / "report.ptm_site_matrix.tsv")
        write_main_report(long_df, out / "report.tsv")
        click.secho("[diaquant] done (no identifications).", fg="yellow")
        return

    # ----- RT prediction error filter -----
    # Remove PSMs whose observed RT deviates too far from Sage's predicted RT.
    # This is a lightweight false-positive filter that does not require
    # re-scoring; Sage generates predicted_rt (→ Predicted.RT) when
    # predict_rt=true.  The tolerance is in minutes; None disables the filter.
    if (
        cfg.rt_prediction_tolerance_min is not None
        and "Predicted.RT" in long_df.columns
        and not long_df.empty
    ):
        rt_err = (long_df["RT"] - long_df["Predicted.RT"]).abs()
        before = len(long_df)
        long_df = long_df[rt_err <= cfg.rt_prediction_tolerance_min]
        click.echo(
            f"[diaquant] RT prediction filter "
            f"(|RT - Predicted.RT| ≤ {cfg.rt_prediction_tolerance_min:.1f} min): "
            f"{before} → {len(long_df)} PSMs"
        )

    # ----- LOWESS run-to-run RT alignment (always-on) -----
    if cfg.rt_alignment:
        click.echo(f"[diaquant] RT alignment (LOWESS frac={cfg.rt_align_frac})")
        long_df, rt_stats = align_runs(
            long_df,
            params=RTAlignParams(
                enabled=True,
                frac=cfg.rt_align_frac,
                min_anchors=cfg.rt_align_min_anchors,
                q_cutoff=cfg.rt_align_q_cutoff,
            ),
        )
        rt_stats_path = cfg.output_dir / "report.rt_alignment.tsv"
        write_rt_stats(rt_stats, rt_stats_path)
        click.echo(f"  wrote {rt_stats_path}")
        # short summary on the console
        if not rt_stats.empty:
            aligned = rt_stats[rt_stats["role"] == "aligned"]
            if not aligned.empty:
                click.echo(
                    "  RMSE drift (sec):  before median = "
                    f"{aligned['rmse_sec_before'].median():.2f}, "
                    f"after median = {aligned['rmse_sec_after'].median():.2f}"
                )
    else:
        click.echo("[diaquant] RT alignment disabled (rt_alignment: false)")

    if long_df.empty:
        # RT filter or RT alignment may have emptied the frame; report it.
        click.secho(
            "[diaquant] PSMs were found but removed by RT filter / alignment. "
            "Try relaxing rt_prediction_tolerance_min.",
            fg="yellow",
        )
        out = cfg.output_dir
        sample_names = [Path(p).name for p in cfg.mzml_files]
        write_pr_matrix(
            pd.DataFrame(columns=PR_COLS + sample_names),
            out / "report.pr_matrix.tsv",
        )
        pd.DataFrame(columns=PG_COLS + sample_names).to_csv(
            out / "report.pg_matrix.tsv", sep="\t", index=False, na_rep=""
        )
        write_site_matrix(pd.DataFrame(), out / "report.ptm_site_matrix.tsv")
        write_main_report(long_df, out / "report.tsv")
        click.secho("[diaquant] done (no identifications).", fg="yellow")
        return

    pr_wide = precursor_matrix(long_df)
    pg_lfq = protein_quant(long_df, min_samples=cfg.quant_min_samples)
    site_lfq = site_quant(long_df, min_samples=cfg.quant_min_samples)

    out = cfg.output_dir
    write_pr_matrix(pr_wide, out / "report.pr_matrix.tsv")
    write_pg_matrix(pg_lfq, pr_wide, out / "report.pg_matrix.tsv")
    write_site_matrix(site_lfq, out / "report.ptm_site_matrix.tsv")
    write_main_report(long_df, out / "report.tsv")

    if cfg.sample_sheet is not None:
        click.echo(f"[diaquant] sample sheet detected: {cfg.sample_sheet}")
        sheet = load_sample_sheet(cfg.sample_sheet)
        try:
            pg_full = (out / "report.pg_matrix.tsv")
            pg_df = __import__("pandas").read_csv(pg_full, sep="\t")
            pg_diff = differential(pg_df, sheet,
                                   id_cols=["Protein.Group", "Protein.Names",
                                            "Genes", "First.Protein.Description",
                                            "N.Sequences", "N.Proteotypic.Sequences"])
            pg_diff.to_csv(out / "report.pg_matrix.diff.tsv",
                           sep="\t", index=False)
            click.echo(f"  wrote {out/'report.pg_matrix.diff.tsv'}")
        except Exception as exc:                           # pragma: no cover
            click.secho(f"  (skipping pg differential: {exc})", fg="yellow")

    click.secho("[diaquant] done.", fg="green")
    click.echo(f"  wrote {out/'report.pr_matrix.tsv'}")
    click.echo(f"  wrote {out/'report.pg_matrix.tsv'}")
    click.echo(f"  wrote {out/'report.ptm_site_matrix.tsv'}")
    click.echo(f"  wrote {out/'report.tsv'}")


if __name__ == "__main__":
    cli()
