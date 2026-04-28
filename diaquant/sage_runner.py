"""Build a Sage JSON configuration from a DiaQuantConfig and run the binary.

Sage (https://github.com/lazear/sage, MIT licence) is the search-engine backbone
of diaquant.  It accepts narrow- or wide-window DIA mzML data, supports
arbitrary variable / fixed modifications via simple mass shifts and computes
target-decoy FDR at PSM, peptide and protein level.

This module is responsible for two things only: (1) translating a
``DiaQuantConfig`` into the JSON schema understood by Sage v0.14 and (2)
invoking the binary, capturing stdout/stderr and returning the path to the
``results.sage.tsv`` file that downstream modules consume.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import DiaQuantConfig
from .modifications import Modification, resolve_modifications


# Sage v0.14 syntax:
#   static_mods    -> Dict[str, float]            scalar value per residue
#   variable_mods  -> Dict[str, List[float]]      list of values per residue
# Termini conventions:
#   '^'  peptide N-term  (variable only)
#   '$'  peptide C-term  (variable only)
#   '['  protein N-term
#   ']'  protein C-term
def _mods_to_sage_static(mods: List[Modification]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in mods:
        if not m.fixed:
            continue
        for residue in m.targets:
            # Sage static_mods accepts only one value per key; warn on conflict
            key = residue
            if key in out and abs(out[key] - m.mass_shift) > 1e-6:
                raise ValueError(
                    f"Multiple static modifications declared for residue '{key}'."
                    f" Sage allows only one fixed mod per residue.")
            out[key] = round(m.mass_shift, 6)
    return out


def _mods_to_sage_variable(mods: List[Modification]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for m in mods:
        if m.fixed:
            continue
        for residue in m.targets:
            out.setdefault(residue, []).append(round(m.mass_shift, 6))
    return out


def build_sage_config(cfg: DiaQuantConfig) -> dict:
    """Translate a DiaQuantConfig into a Sage v0.14 JSON parameter object."""
    mods = resolve_modifications(
        list(cfg.fixed_modifications) + list(cfg.variable_modifications),
        cfg.custom_modifications,
    )
    fixed_mods = _mods_to_sage_static(mods)
    var_mods = _mods_to_sage_variable(mods)

    # Convert peptide-charge range to peptide mass range (Sage filters by mass).
    # Average peptide is ≈ ((m/z * z) - z*proton).  We use the precursor m/z range
    # × charge bounds to compute conservative peptide mass cut-offs.
    proton = 1.007276
    peptide_min_mass = max(
        500.0,
        cfg.min_precursor_mz * cfg.min_precursor_charge
        - cfg.min_precursor_charge * proton,
    )
    peptide_max_mass = min(
        6000.0,
        cfg.max_precursor_mz * cfg.max_precursor_charge
        - cfg.max_precursor_charge * proton,
    )

    sage = {
        "database": {
            "bucket_size": 32768,
            "enzyme": {
                "missed_cleavages": cfg.missed_cleavages,
                "min_len": cfg.min_peptide_length,
                "max_len": cfg.max_peptide_length,
                "cleave_at": "KR" if cfg.enzyme == "trypsin" else (
                    "K" if cfg.enzyme == "lys-c" else "$"),
                "restrict": "P" if cfg.enzyme == "trypsin" else None,
            },
            "fragment_min_mz": cfg.min_fragment_mz,
            "fragment_max_mz": cfg.max_fragment_mz,
            "peptide_min_mass": peptide_min_mass,
            "peptide_max_mass": peptide_max_mass,
            "ion_kinds": ["b", "y"],
            "min_ion_index": 2,
            "static_mods": fixed_mods,
            "variable_mods": var_mods,
            "max_variable_mods": cfg.max_variable_mods,
            "decoy_tag": "rev_",
            "generate_decoys": True,
            "fasta": str(cfg.fasta),
        },
        # DIA scoring strategy:
        # - lfq=false: use DDA-style scoring to avoid the MS1-feature-tracing
        #   bootstrap failure that silently empties results.sage.tsv with 1-2 files.
        # - wide_window=true: recommended Sage mode for DIA; relaxes precursor m/z
        #   filtering and primarily scores via fragment ion matching (better for
        #   chimeric spectra in narrow/medium-window DIA).
        "quant": {
            "lfq": False,
        },
        "precursor_tol":  {"ppm": [-cfg.precursor_tol_ppm, cfg.precursor_tol_ppm]},
        "fragment_tol":   {"ppm": [-cfg.fragment_tol_ppm,  cfg.fragment_tol_ppm]},
        "precursor_charge": [cfg.min_precursor_charge, cfg.max_precursor_charge],
        "isotope_errors": list(cfg.isotope_errors),
        "deisotope": True,
        "chimera": True,
        "wide_window": True,               # DIA mode: score via fragments, relax precursor m/z
        "predict_rt": True,
        "min_peaks": 15,
        "max_peaks": 150,
        "min_matched_peaks": 3,            # permissive for chimeric DIA spectra
        "report_psms": 1,
        "max_fragment_charge": 2,
        "output_directory": str(cfg.output_dir / "sage"),
        "mzml_paths": [str(p) for p in cfg.mzml_files],
    }
    return sage


def run_sage(cfg: DiaQuantConfig) -> Path:
    """Write the Sage JSON config to disk and execute the binary.

    Returns the path to the ``results.sage.tsv`` produced by Sage.  Raises
    ``subprocess.CalledProcessError`` if Sage returns non-zero.
    """
    sage_cfg = build_sage_config(cfg)
    out_dir = Path(sage_cfg["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "sage_config.json"
    with open(cfg_path, "w") as fh:
        json.dump(sage_cfg, fh, indent=2)

    cmd = [cfg.sage_binary, str(cfg_path)]
    if cfg.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(cfg.threads)

    print(f"[diaquant] Running Sage: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    results = out_dir / "results.sage.tsv"
    if not results.exists():
        raise FileNotFoundError(
            f"Sage finished but {results} was not produced. "
            "Check the Sage stdout above."
        )
    return results


def run_sage_batched(cfg: DiaQuantConfig) -> Path:
    """Run Sage in memory-safe batches and merge results.

    When ``cfg.batch_size > 0`` and there are more files than ``batch_size``,
    Sage is executed ``ceil(n / batch_size)`` times, each time on a subset of
    the mzML files.  The ``results.sage.tsv`` files are concatenated and a
    merged TSV is written to the normal output location.

    When batching is not needed (batch_size==0 or files fit in one batch) this
    falls back to a normal ``run_sage()`` call.
    """
    all_files = list(cfg.mzml_files)
    n = len(all_files)
    bs = cfg.batch_size

    if bs <= 0 or bs >= n:
        return run_sage(cfg)

    batches = [all_files[i: i + bs] for i in range(0, n, bs)]
    n_batches = len(batches)
    print(f"[diaquant] auto-batch: {n} files → {n_batches} batches of ≤{bs} files")

    merged_dir = Path(cfg.output_dir) / "sage"
    merged_dir.mkdir(parents=True, exist_ok=True)

    result_tsvs: List[Path] = []
    for idx, batch in enumerate(batches):
        batch_cfg = replace(cfg)
        batch_cfg.mzml_files = batch
        batch_cfg.output_dir = Path(cfg.output_dir) / f"sage_batch_{idx}"
        print(f"[diaquant] batch {idx + 1}/{n_batches}: {[p.name for p in batch]}")
        result_tsvs.append(run_sage(batch_cfg))

    # Merge all results.sage.tsv → single file in the canonical sage/ dir
    print(f"[diaquant] merging {n_batches} batch results...")
    dfs = [pd.read_csv(tsv, sep="\t") for tsv in result_tsvs]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_path = merged_dir / "results.sage.tsv"
    merged_df.to_csv(merged_path, sep="\t", index=False)

    # Also create a stub sage_config.json pointing at the merged output
    # so downstream code (parse_sage_tsv) can find it
    stub_config = {"batches": n_batches, "total_files": n, "batch_size": bs}
    (merged_dir / "sage_config.json").write_text(
        json.dumps(stub_config, indent=2)
    )
    print(f"[diaquant] merged {len(merged_df)} PSMs from {n_batches} batches → {merged_path}")
    return merged_path
