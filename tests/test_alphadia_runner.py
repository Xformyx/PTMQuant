"""Phase 2 unit tests for ``diaquant.alphadia_runner.build_alphadia_config``.

These tests exercise the PTM-aware mapping introduced in v0.6.0a3:

  1. Each PTMQuant ``PassProfile`` produces an AlphaDIA YAML dict that
     declares the right ``library_prediction.variable_modifications``,
     ``fixed_modifications``, FDR cutoff, and PeptDeep model type.
  2. ``Acetyl_Nterm`` is translated to ``Acetyl@Protein_N-term``.
  3. ``Phospho`` (targets ``("S","T","Y")``) expands across all three
     residues.
  4. ``library_path`` flips ``library_prediction.enabled`` to False.
  5. Per-pass numeric overrides (``missed_cleavages``,
     ``max_variable_mods``, ``site_probability_cutoff`` indirectly via
     ``peptide_fdr``) override the global ``DiaQuantConfig`` defaults.
  6. PTMQuant audit metadata is paired with the AlphaDIA dict (not embedded
     — AlphaDIA 2.x rejects unknown top-level YAML keys) and is persisted
     by ``run_alphadia`` as ``ptmquant_alphadia_meta.json``.

Tests follow the existing ``tests/test_smoke.py`` style: plain pytest
functions, no external fixtures, no AlphaDIA installation required
(``build_alphadia_config`` is a pure-Python translator).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from diaquant.alphadia_runner import _alphadia_config_and_audit, build_alphadia_config
from diaquant.config import DiaQuantConfig
from diaquant.ptm_profiles import PASS_PROFILES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path) -> DiaQuantConfig:
    """Minimal DiaQuantConfig good enough to exercise the translator."""
    fasta = tmp_path / "proteome.fasta"
    fasta.write_text(">SEED\nMSEED\n")
    mzml = tmp_path / "sample.mzML"
    mzml.write_text("<mzML/>")
    return DiaQuantConfig(
        fasta=fasta,
        mzml_files=[mzml],
        output_dir=tmp_path / "out",
    )


# ---------------------------------------------------------------------------
# Whole proteome (backbone) pass
# ---------------------------------------------------------------------------

def test_whole_proteome_pass_uses_global_fdr_and_generic_model(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out, audit = _alphadia_config_and_audit(cfg, pass_profile=PASS_PROFILES["whole_proteome"])
    assert all(not k.startswith("_") for k in out)
    assert out["fdr"]["fdr"] == pytest.approx(0.01)
    # No specialised PeptDeep model for the backbone pass.
    assert out["library_prediction"]["peptdeep_model_type"] == "generic"
    # Variable modifications: only the always-on M-Ox + protein N-term Ac.
    var = out["library_prediction"]["variable_modifications"]
    assert "Oxidation@M" in var
    assert "Acetyl@Protein_N-term" in var
    # No PTM-specific tokens leak in.
    for forbidden in ("Phospho@", "GlyGly@", "Methyl@", "Succinyl@"):
        assert forbidden not in var
    # Fixed mod is the global Carbamidomethyl@C.
    assert out["library_prediction"]["fixed_modifications"] == "Carbamidomethyl@C"
    # Inference defaults to gene-level, matching PTMQuant downstream.
    assert out["fdr"]["group_level"] == "genes"
    # Sidecar audit records the resolved tokens.
    assert audit["pass_name"] == "whole_proteome"
    assert audit["pass_is_whole_proteome"] is True
    assert audit["ptmquant_fdr_source"] == "cfg.peptide_fdr"
    assert audit["ptmquant_peptdeep_model_type"] == "generic"


# ---------------------------------------------------------------------------
# Phospho pass: STY expansion + 5% FDR + phospho PeptDeep model
# ---------------------------------------------------------------------------

def test_phospho_pass_expands_sty_and_relaxes_fdr(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out, audit = _alphadia_config_and_audit(cfg, pass_profile=PASS_PROFILES["phospho"])

    # 5% peptide FDR (Bekker-Jensen 2020 phospho-DIA convention) maps onto
    # AlphaDIA's single fdr.fdr cutoff.
    assert out["fdr"]["fdr"] == pytest.approx(0.05)
    # Phospho-aware PeptDeep model.
    assert out["library_prediction"]["peptdeep_model_type"] == "phospho"
    # STY expansion: every residue gets its own token.
    var = out["library_prediction"]["variable_modifications"]
    assert "Phospho@S" in var
    assert "Phospho@T" in var
    assert "Phospho@Y" in var
    # Pass override: 3 variable mods (vs global default 2).
    assert out["library_prediction"]["max_var_mod_num"] == 3
    # Audit confirms the FDR came from the pass profile.
    assert audit["ptmquant_fdr_source"] == "pass_profile.peptide_fdr"


# ---------------------------------------------------------------------------
# Ubiquitin (K-GG) pass: digly model + missed_cleavages=3
# ---------------------------------------------------------------------------

def test_ubiquitin_pass_uses_digly_model_and_three_missed_cleavages(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = build_alphadia_config(cfg, pass_profile=PASS_PROFILES["ubiquitin"])

    assert out["library_prediction"]["peptdeep_model_type"] == "digly"
    # K-GG blocks tryptic cleavage at the modified lysine — bump to 3.
    assert out["library_prediction"]["missed_cleavages"] == 3
    var = out["library_prediction"]["variable_modifications"]
    assert "GlyGly@K" in var
    # Same 5% FDR convention as other PTM passes.
    assert out["fdr"]["fdr"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Acetyl/methyl pass: many K/R tokens, 5% FDR
# ---------------------------------------------------------------------------

def test_acetyl_methyl_pass_emits_kr_tokens(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = build_alphadia_config(cfg, pass_profile=PASS_PROFILES["acetyl_methyl"])

    var = out["library_prediction"]["variable_modifications"]
    # Acetyl on K (variable, separate from the protein-N-term variant).
    assert "Acetyl@K" in var
    # Methyl/Dimethyl expand across K and R.
    for token in (
        "Methyl@K", "Methyl@R",
        "Dimethyl@K", "Dimethyl@R",
        "Trimethyl@K",
    ):
        assert token in var, f"missing {token!r} in {var!r}"
    # Falls back to generic PeptDeep model (no specialised KAc model in
    # AlphaDIA 2.1.1).
    assert out["library_prediction"]["peptdeep_model_type"] == "generic"


# ---------------------------------------------------------------------------
# Library path toggle: precomputed library disables in-engine prediction
# ---------------------------------------------------------------------------

def test_library_path_disables_library_prediction(tmp_path: Path):
    cfg = _cfg(tmp_path)
    lib = tmp_path / "predicted_library.tsv"
    lib.write_text("modified_sequence\tcharge\nPEPTIDER\t2\n")

    out, audit = _alphadia_config_and_audit(
        cfg,
        pass_profile=PASS_PROFILES["whole_proteome"],
        library_path=lib,
    )

    assert out["library_path"] == str(lib.resolve())
    assert out["library_prediction"]["enabled"] is False
    assert audit["ptmquant_library_mode"] == "precomputed"


def test_no_library_path_enables_in_engine_prediction(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out, audit = _alphadia_config_and_audit(cfg, pass_profile=PASS_PROFILES["whole_proteome"])
    assert out["library_path"] is None
    assert out["library_prediction"]["enabled"] is True
    assert audit["ptmquant_library_mode"] == "in_engine_prediction"


# ---------------------------------------------------------------------------
# Pass-less call (single-pass legacy run): falls back to cfg.peptide_fdr
# ---------------------------------------------------------------------------

def test_no_pass_profile_falls_back_to_cfg(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out, audit = _alphadia_config_and_audit(cfg, pass_profile=None)
    # cfg.peptide_fdr default is 0.01.
    assert out["fdr"]["fdr"] == pytest.approx(0.01)
    assert out["library_prediction"]["peptdeep_model_type"] == "generic"
    # Falls back to global cfg.variable_modifications (Oxidation, Acetyl_Nterm).
    var = out["library_prediction"]["variable_modifications"]
    assert "Oxidation@M" in var
    assert "Acetyl@Protein_N-term" in var
    # Audit records the fallback source.
    assert audit["ptmquant_fdr_source"] == "cfg.peptide_fdr"
    assert audit["pass_name"] is None


# ---------------------------------------------------------------------------
# Per-pass fragment tolerance override (oglcnac pass relaxes to 15.0 ppm)
# ---------------------------------------------------------------------------

def test_oglcnac_pass_relaxes_fragment_tolerance(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = build_alphadia_config(cfg, pass_profile=PASS_PROFILES["oglcnac"])
    # cfg.fragment_tol_ppm default is 12.0; pass overrides to 15.0.
    assert out["search"]["target_ms2_tolerance"] == pytest.approx(15.0)
    var = out["library_prediction"]["variable_modifications"]
    assert "OGlcNAc@S" in var
    assert "OGlcNAc@T" in var


# ---------------------------------------------------------------------------
# Sanity: every built-in pass produces a valid, JSON-serialisable dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pass_name", list(PASS_PROFILES.keys()))
def test_all_builtin_passes_produce_valid_alphadia_config(
    pass_name: str, tmp_path: Path
):
    import json

    cfg = _cfg(tmp_path)
    out = build_alphadia_config(cfg, pass_profile=PASS_PROFILES[pass_name])

    assert "_ptmquant" not in out
    assert not any(k.startswith("_") for k in out), f"{pass_name}: underscore top-level keys forbidden for AlphaDIA 2.x"

    # Required AlphaDIA top-level keys.
    for key in ("raw_paths", "output_directory", "library_prediction",
                "search", "fdr"):
        assert key in out, f"{pass_name}: missing top-level key {key!r}"

    # Required library_prediction keys.
    lp = out["library_prediction"]
    for key in ("enabled", "variable_modifications", "fixed_modifications",
                "max_var_mod_num", "missed_cleavages", "precursor_len",
                "precursor_charge", "peptdeep_model_type"):
        assert key in lp, f"{pass_name}: missing library_prediction.{key}"

    # FDR is a float in [0, 1].
    assert 0.0 < out["fdr"]["fdr"] <= 0.05, (
        f"{pass_name}: fdr={out['fdr']['fdr']!r} out of expected range"
    )

    # Must be JSON-serialisable so we can write it as YAML at runtime.
    json.dumps(out)


def test_run_alphadia_writes_only_alphadia_keys_in_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaDIA 2.x merges user YAML onto defaults — unknown keys -> CONFIG_ERROR."""
    import json
    import subprocess as sp

    from diaquant.alphadia_runner import (
        AlphaDIAProbe,
        run_alphadia,
    )

    monkeypatch.setattr(
        "diaquant.alphadia_runner.probe_alphadia",
        lambda **kw: AlphaDIAProbe(True, "/fake/bin/alphadia", "2.1.1", None),
    )
    monkeypatch.setattr(sp, "run", lambda *_a, **_k: sp.CompletedProcess([], 0))

    cfg = _cfg(tmp_path)
    out_dir = tmp_path / "alphadia_pass"
    run_alphadia(cfg, pass_profile=PASS_PROFILES["phospho"], output_dir=out_dir)

    loaded = json.loads((out_dir / "alphadia_config.yaml").read_text())
    assert "_ptmquant" not in loaded
    assert all(not str(k).startswith("_") for k in loaded)

    meta = json.loads((out_dir / "ptmquant_alphadia_meta.json").read_text())
    assert meta["pass_name"] == "phospho"
