"""Smoke tests: exercise pure-Python modules without needing real mzML files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from diaquant.modifications import (
    DEFAULT_MODIFICATIONS,
    parse_user_modifications,
    resolve_modifications,
)
from diaquant.ptm_localization import add_site_probabilities, extract_mod_positions
from diaquant.sage_runner import build_sage_config, _mods_to_sage_static, _mods_to_sage_variable
from diaquant.config import DiaQuantConfig
from diaquant.ptm_profiles import PASS_PROFILES, resolve_passes
from diaquant.multipass import _config_for_pass
from diaquant.stats import _bh_fdr


def test_default_modifications_complete():
    must_have = {"Phospho", "GlyGly", "Acetyl", "Methyl", "Dimethyl",
                 "Trimethyl", "Succinyl", "Malonyl", "Crotonyl", "Carbamidomethyl"}
    assert must_have <= set(DEFAULT_MODIFICATIONS)


def test_resolve_modifications_handles_custom():
    extra = [{"name": "Lactyl", "unimod_id": 2114, "mass_shift": 72.021129,
              "targets": ["K"]}]
    mods = resolve_modifications(["Phospho", "Acetyl"], extra)
    names = {m.name for m in mods}
    assert names == {"Phospho", "Acetyl", "Lactyl"}


def test_extract_mod_positions_diann_syntax():
    seq = "AAAS(UniMod:21)TR"
    assert extract_mod_positions(seq) == [(4, "UniMod:21")]


def test_extract_mod_positions_sage_syntax():
    seq = "AAAS[+79.966331]TR"
    assert extract_mod_positions(seq) == [(4, "+79.966331")]


def test_add_site_probabilities_softmax():
    psm = pd.DataFrame({
        "filename": ["a.mzML"] * 3,
        "peptide": ["AAS[+79.966331]TR", "AAST[+79.966331]R", "AAS[+79.966331]T[+79.966331]R"],
        "charge": [2, 2, 2],
        "hyperscore": [30.0, 25.0, 10.0],
    })
    out = add_site_probabilities(psm, cutoff=0.5)
    # the highest hyperscore should win the softmax
    best = out.loc[out["hyperscore"].idxmax()]
    assert best["Best.Site.Probability"] > 0.95
    assert best["PTM.Site.Confident"]


def test_build_sage_config_minimal(tmp_path: Path):
    cfg = DiaQuantConfig(
        fasta=tmp_path / "x.fasta",
        mzml_files=[tmp_path / "a.mzML"],
        output_dir=tmp_path / "out",
        variable_modifications=["Phospho", "Methyl"],
    )
    sage_cfg = build_sage_config(cfg)
    var = sage_cfg["database"]["variable_mods"]
    assert "S" in var and 79.966331 in var["S"]
    assert "K" in var and 14.01565 in var["K"]
    assert sage_cfg["chimera"] is True
    assert sage_cfg["database"]["generate_decoys"] is True


def test_resolve_passes_picks_only_selected():
    profiles = resolve_passes(["whole_proteome", "phospho"])
    assert [p.name for p in profiles] == ["whole_proteome", "phospho"]
    assert "acetyl_methyl" not in [p.name for p in profiles]


def test_resolve_passes_rejects_unknown():
    with pytest.raises(KeyError):
        resolve_passes(["nonexistent_pass"])


def test_resolve_passes_empty_raises():
    with pytest.raises(ValueError):
        resolve_passes([])


def test_config_for_pass_overrides_correctly(tmp_path: Path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">sp|P00001|TEST_HUMAN test\nMAAAAA\n")
    mzml = tmp_path / "y.mzML"
    mzml.write_text("<placeholder/>")
    base = DiaQuantConfig(
        fasta=fasta, mzml_files=[mzml],
        output_dir=tmp_path / "out",
        missed_cleavages=2, max_variable_mods=2,
    )
    base.output_dir.mkdir(exist_ok=True)
    profile = PASS_PROFILES["ubiquitin"]
    pcfg = _config_for_pass(base, profile)
    assert pcfg.missed_cleavages == 3
    assert "GlyGly" in pcfg.variable_modifications
    assert pcfg.output_dir.name == "pass_ubiquitin"
    assert base.missed_cleavages == 2  # base unchanged


def test_bh_fdr_monotonic():
    pvals = [0.001, 0.01, 0.04, 0.5, 0.9]
    q = _bh_fdr(pvals)
    assert all(qi >= pi - 1e-9 for qi, pi in zip(q, pvals))
    assert all(q[i] <= q[i+1] + 1e-9 for i in range(len(q)-1))
