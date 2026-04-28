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

    # ---- instrument preset (NEW in 0.5.1) ----
    # Selects a family of numeric defaults (tolerances, m/z range, NCE,
    # AlphaPeptDeep instrument tag) appropriate for the listed Orbitrap
    # generation.  Accepted values: 'exploris_240' (default, matches 0.4.x),
    # 'orbitrap_astral', 'orbitrap_eclipse', 'fusion_lumos'.  A value that the
    # user sets explicitly in the YAML *always* wins over the preset.
    instrument: str = "exploris_240"

    # ---- enzyme / digestion (DIA-NN parlance: "Protease", "Missed cleavages") ----
    # Full catalog lives in :mod:`diaquant.enzymes`.  Accepted values include
    # 'trypsin' (= Trypsin/P, default), 'trypsin-strict', 'lys-c', 'lys-c-strict',
    # 'arg-c', 'asp-n', 'glu-c', 'chymotrypsin' and 'no-cleavage'.
    enzyme: str = "trypsin"
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

    # ---- AlphaPeptDeep predicted spectral library (NEW in 0.5.0) ----
    # When enabled, diaquant uses AlphaPeptDeep to pre-compute a predicted
    # spectral library (RT + MS2 intensity per fragment) for every pass and
    # then post-hoc re-scores the Sage PSMs with two extra features:
    #    * pred_rt_delta   (observed RT vs predicted)
    #    * frag_cosine     (cosine similarity of observed vs predicted MS2)
    # This raises identification depth for all PTMs that DIA-NN cannot
    # handle natively (acetyl, mono/di/tri-methyl, succinyl, malonyl,
    # crotonyl, SUMO QQTGG remnants, citrullination, O-GlcNAc, lactyl, ...).
    predicted_library: bool = True            # default ON per user request
    pred_lib_instrument: str = "QE"          # AlphaPeptDeep instrument preset;
                                              # 'QE' is the right choice for an
                                              # Orbitrap Exploris 240.  Other
                                              # supported values include
                                              # 'Lumos', 'timsTOF', 'SciexTOF',
                                              # 'ThermoTOF'.
    pred_lib_nce: float = 27.0                # normalized collision energy
                                              # (HCD).  Exploris 240 default ≈ 27.
    pred_lib_model_dir: Optional[Path] = None  # override AlphaPeptDeep model
                                              # cache dir.  Default: standard
                                              # ~/peptdeep/pretrained_models
                                              # installed by
                                              # `peptdeep install-models`.
    pred_lib_cache: bool = True               # cache the predicted library
                                              # on disk and skip re-prediction
                                              # on subsequent runs with the
                                              # same FASTA + pass parameters.
    pred_lib_transfer_learning: bool = False  # opt-in: after pass 1, fine-tune
                                              # RT/MS2 on the user's own
                                              # high-confidence PSMs before
                                              # predicting subsequent passes.
    pred_lib_transfer_epochs: int = 10        # epochs for the opt-in fine-tune.
    pred_lib_fallback_in_silico: bool = True  # if AlphaPeptDeep prediction
                                              # fails, fall back to Sage's
                                              # built-in theoretical library
                                              # rather than aborting the run.
    # ---- Post-hoc PSM rescoring (NEW in 0.5.0) ----
    rescore_with_prediction: bool = True      # use the predicted library to
                                              # add pred_rt_delta and
                                              # frag_cosine features to every
                                              # Sage PSM and re-compute FDR.
    rescore_rt_tol_min: float = 3.0           # |observed RT - predicted RT|
                                              # cap (minutes).  PSMs beyond
                                              # this are demoted, not dropped.
    rescore_frag_cosine_cutoff: float = 0.0   # minimum fragment cosine to
                                              # retain a PSM (0 = disabled).

    # ---- run-to-run RT alignment (LOWESS) ----
    rt_alignment: bool = True               # always-on by default per user request
    rt_align_frac: float = 0.2              # LOWESS smoothing fraction
    rt_align_min_anchors: int = 50          # min common PSMs to align a run
    rt_align_q_cutoff: float = 0.01         # PSM q-value upper bound for anchors (ref + per-run)

    # ---- RT prediction filter ----
    # After Sage search, Sage predicts an expected RT for every PSM
    # (predict_rt: true).  PSMs whose |observed_RT - predicted_RT| exceeds
    # this threshold (in minutes) are discarded as likely false positives.
    # Set to None (or omit from YAML) to disable.
    # Typical value: 5.0 min for a 90-min LC gradient.
    rt_prediction_tolerance_min: Optional[float] = 5.0

    # ---- runtime ----
    threads: int = 0                        # 0 = autodetect
    sage_binary: str = "sage"

    # ---- auto-batching ----
    # When > 0, Sage is run ceil(n_files / batch_size) times and results are
    # merged before downstream analysis.  Use this when RAM is insufficient to
    # load all mzML files at once.  0 = no batching (all files in one run).
    batch_size: int = 0

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
        # Resolve instrument preset *after* the user's YAML values have been
        # applied so that explicit user values still win.  apply_preset only
        # overwrites fields whose value still equals the 0.4.x default.
        from .instruments import apply_preset, get_instrument
        apply_preset(cfg, get_instrument(cfg.instrument))
        # Validate the enzyme eagerly so the user sees a clear error instead
        # of a cryptic Sage failure later in the pipeline.
        from .enzymes import get_enzyme
        get_enzyme(cfg.enzyme)
        # Coerce sample_sheet to Path if provided.
        if cfg.sample_sheet is not None and not isinstance(cfg.sample_sheet, Path):
            cfg.sample_sheet = Path(cfg.sample_sheet)
            if not cfg.sample_sheet.exists():
                raise FileNotFoundError(
                    f"sample_sheet not found: {cfg.sample_sheet}"
                )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return cfg
