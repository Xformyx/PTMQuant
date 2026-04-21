"""Load and validate the diaquant YAML configuration file.

The schema mirrors the DIA-NN settings the user is already familiar with so
that diaquant can be a drop-in replacement.  Defaults are tuned for a
**Thermo Orbitrap Exploris 240 / Astral / Eclipse / Lumos** running narrow
DIA windows (≤ 10 m/z), which is the configuration the user actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class DiaQuantConfig:
    fasta: Path
    mzml_files: List[Path]
    output_dir: Path

    # ---- enzyme / digestion (DIA-NN parlance: "Protease", "Missed cleavages") ----
    enzyme: str = "trypsin"                 # 'trypsin' (=Trypsin/P), 'lys-c', 'no-cleavage'
    missed_cleavages: int = 2
    min_peptide_length: int = 7
    max_peptide_length: int = 30

    # ---- precursor / fragment ranges (DIA-NN: "Precursor m/z", "Fragment ion m/z") ----
    min_precursor_charge: int = 2
    max_precursor_charge: int = 4
    min_precursor_mz: float = 400.0         # 6 m/z DIA on Exploris starts at ≈ 403
    max_precursor_mz: float = 1000.0
    min_fragment_mz: float = 200.0
    max_fragment_mz: float = 1500.0

    # ---- tolerances (DIA-NN: "MS1 accuracy", "MS2 accuracy") ----
    precursor_tol_ppm: float = 6.0          # identification tolerance (Exploris/Astral default)
    fragment_tol_ppm: float = 12.0
    library_precursor_tol_ppm: float = 8.0  # library generation tolerance (slightly looser)
    library_fragment_tol_ppm: float = 15.0
    isotope_errors: Tuple[int, int] = (-1, 3)

    # ---- modifications ----
    fixed_modifications: List[str] = field(default_factory=lambda: ["Carbamidomethyl"])
    variable_modifications: List[str] = field(default_factory=lambda: ["Oxidation", "Acetyl_Nterm"])
    custom_modifications: List[dict] = field(default_factory=list)
    max_variable_mods: int = 2              # DIA-NN identification default

    # ---- multi-pass PTM search ----
    # If non-empty, the run executes one Sage search per pass and merges
    # results.  Each pass overrides the global modification settings.
    # When the list is empty diaquant falls back to a single-pass run that
    # uses fixed_modifications / variable_modifications above.
    passes: List[str] = field(default_factory=list)
    custom_passes: List[dict] = field(default_factory=list)

    # ---- sample sheet (optional) ----
    # Path to a TSV with at least columns ``mzml_file`` and ``group``.  If
    # provided, condition×timepoint statistics are computed automatically.
    sample_sheet: Optional[Path] = None

    # ---- FDR & site localization ----
    psm_fdr: float = 0.01
    peptide_fdr: float = 0.01
    protein_fdr: float = 0.01
    site_probability_cutoff: float = 0.75
    scoring_mode: str = "peptidoforms"      # 'peptidoforms' | 'standard'  (DIA-NN: Scoring)

    # ---- quantification (DIA-NN: "QuantUMS", "MBR") ----
    quant_top_n_fragments: int = 6
    quant_min_samples: int = 2
    match_between_runs: bool = True         # DIA-NN MBR
    machine_learning: str = "nn_cv"         # 'nn_cv' | 'linear'  (DIA-NN: NNs cross-validated)

    # ---- DIA acquisition ----
    wide_window: bool = False               # True for SWATH / wide-window DIA (>10 m/z)

    # ---- run-to-run RT alignment (LOWESS) ----
    rt_alignment: bool = True               # always-on by default per user request
    rt_align_frac: float = 0.2              # LOWESS smoothing fraction
    rt_align_min_anchors: int = 50          # min common PSMs to align a run
    rt_align_q_cutoff: float = 0.01         # PSM q-value upper bound for anchors

    # ---- runtime ----
    threads: int = 0                        # 0 = autodetect
    sage_binary: str = "sage"

    @staticmethod
    def from_yaml(path: str | Path) -> "DiaQuantConfig":
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh)

        mzml = [Path(p) for p in data["mzml_files"]]
        for p in mzml:
            if not p.exists():
                raise FileNotFoundError(f"mzML file not found: {p}")
        fasta = Path(data["fasta"])
        if not fasta.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta}")

        cfg = DiaQuantConfig(
            fasta=fasta,
            mzml_files=mzml,
            output_dir=Path(data.get("output_dir", "diaquant_results")),
        )
        for k, v in data.items():
            if k in {"fasta", "mzml_files", "output_dir"}:
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # Coerce sample_sheet to Path if provided.
        if cfg.sample_sheet is not None and not isinstance(cfg.sample_sheet, Path):
            cfg.sample_sheet = Path(cfg.sample_sheet)
            if not cfg.sample_sheet.exists():
                raise FileNotFoundError(
                    f"sample_sheet not found: {cfg.sample_sheet}"
                )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return cfg
