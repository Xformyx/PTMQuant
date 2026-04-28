"""Smoke tests: exercise pure-Python modules without needing real mzML files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from diaquant.rt_align import RTAlignParams, align_runs


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


def test_add_site_probabilities_uses_score_after_diann_rename():
    """parse_sage renames Sage ``hyperscore`` → ``Score``; localization must still work."""
    psm = pd.DataFrame({
        "filename": ["a.mzML", "a.mzML"],
        "peptide": ["AAS[+79.966331]TR", "AAST[+79.966331]R"],
        "charge": [2, 2],
        "Score": [30.0, 25.0],
    })
    out = add_site_probabilities(psm, cutoff=0.5)
    assert out["Best.Site.Probability"].max() > 0.99


def test_add_site_probabilities_empty():
    psm = pd.DataFrame({"peptide": pd.Series([], dtype=object)})
    out = add_site_probabilities(psm, cutoff=0.75)
    assert out.empty
    assert "Best.Site.Probability" in out.columns
    assert "PTM.Site.Confident" in out.columns


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


def _make_rt_dataset(n_anchors: int = 200,
                    drift_sec: float = 25.0,
                    seed: int = 0) -> pd.DataFrame:
    """Synthetic two-run dataset for RT alignment tests.

    Run B is shifted by ``drift_sec`` seconds plus mild non-linear curvature
    plus Gaussian noise (sigma = 3 sec).  Run A is the reference.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    rt_ref_min = np.linspace(5.0, 95.0, n_anchors)              # 5..95 minutes
    drift_min = drift_sec / 60.0
    curvature = 0.05 * np.sin(rt_ref_min / 15.0)                # +/- 3 sec wave
    rt_runB_min = rt_ref_min + drift_min + curvature \
                  + rng.normal(0, 3.0 / 60.0, n_anchors)
    peptides = [f"PEP{i:04d}" for i in range(n_anchors)]
    df = pd.DataFrame({
        "filename": ["runA.mzML"] * n_anchors + ["runB.mzML"] * n_anchors,
        "Modified.Sequence": peptides + peptides,
        "Precursor.Charge": [2] * (2 * n_anchors),
        "RT": list(rt_ref_min) + list(rt_runB_min),
        "Q.Value": [0.001] * (2 * n_anchors),
    })
    return df


def test_rt_align_reduces_drift():
    df = _make_rt_dataset()
    aligned, stats = align_runs(df, RTAlignParams())
    runB = stats[stats["filename"] == "runB.mzML"].iloc[0]
    assert runB["role"] == "aligned"
    # the LOWESS fit must shrink the residual RMSE substantially
    assert runB["rmse_sec_after"] < runB["rmse_sec_before"] * 0.5
    # output must contain the new column with finite values for run B
    rB = aligned[aligned["filename"] == "runB.mzML"]
    assert rB["RT.Aligned"].notna().all()
    assert (rB["RT.Aligned"] != rB["RT"]).any()


def test_rt_align_skips_when_few_anchors():
    df = _make_rt_dataset(n_anchors=10)
    _, stats = align_runs(df, RTAlignParams(min_anchors=50))
    runB = stats[stats["filename"] == "runB.mzML"].iloc[0]
    assert runB["role"].startswith("skipped")


def test_rt_align_disabled_returns_unchanged_rt():
    df = _make_rt_dataset()
    aligned, stats = align_runs(df, RTAlignParams(enabled=False))
    # RT.Aligned must equal RT exactly when alignment is off
    assert (aligned["RT.Aligned"] == aligned["RT"]).all()
    assert stats.empty


def test_rt_align_uses_only_confident_anchors():
    """Anchors must be filtered to Q.Value ≤ q_cutoff; noisy rows are excluded."""
    df = _make_rt_dataset(n_anchors=200)
    # poison half the runB rows with Q.Value > cutoff and huge RT offset
    rng = np.random.default_rng(42)
    noise_idx = df[df["filename"] == "runB.mzML"].sample(100, random_state=42).index
    df.loc[noise_idx, "Q.Value"] = 0.5        # above cutoff
    df.loc[noise_idx, "RT"] += rng.uniform(30, 60, size=100)  # big wrong offsets

    aligned_strict, stats_strict = align_runs(df, RTAlignParams(q_cutoff=0.01))
    aligned_loose,  stats_loose  = align_runs(df, RTAlignParams(q_cutoff=1.0))

    runB_strict = stats_strict[stats_strict["filename"] == "runB.mzML"].iloc[0]
    runB_loose  = stats_loose [stats_loose ["filename"] == "runB.mzML"].iloc[0]

    # strict q_cutoff → fewer anchors but better (lower) post-alignment RMSE
    assert runB_strict["n_anchors"] <= runB_loose["n_anchors"]
    # strict must not be dramatically worse than loose (poison rows hurt the loose fit)
    assert runB_strict["rmse_sec_after"] < runB_loose["rmse_sec_after"] * 2


def test_rt_prediction_filter_removes_outliers():
    """PSMs with |RT - Predicted.RT| > tolerance should be dropped."""
    df = pd.DataFrame({
        "filename": ["a.mzML"] * 5,
        "peptide": [f"PEP{i}[+79.966331]R" for i in range(5)],
        "charge": [2] * 5,
        "Score": [30.0] * 5,
        "RT": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Predicted.RT": [10.1, 20.2, 30.3, 30.0, 10.0],  # last two are outliers
        "Peptide.Q.Value": [0.005] * 5,
    })
    tol = 5.0  # minutes
    rt_err = (df["RT"] - df["Predicted.RT"]).abs()
    filtered = df[rt_err <= tol]
    # rows 3 (|40-30|=10) and 4 (|50-10|=40) exceed the 5 min tolerance
    assert len(filtered) == 3
    assert 3 not in filtered.index
    assert 4 not in filtered.index


def test_multipass_peptide_fdr_is_respected(tmp_path):
    """_config_for_pass must propagate peptide_fdr so parse_sage_tsv uses it."""
    from diaquant.config import DiaQuantConfig
    from diaquant.multipass import _config_for_pass
    from diaquant.ptm_profiles import PASS_PROFILES

    fasta = tmp_path / "x.fasta"
    fasta.write_text(">sp|P00001|TEST_HUMAN test\nMAAAAA\n")
    mzml = tmp_path / "y.mzML"
    mzml.write_text("<placeholder/>")
    base = DiaQuantConfig(
        fasta=fasta, mzml_files=[mzml],
        output_dir=tmp_path / "out",
        peptide_fdr=0.05,
    )
    base.output_dir.mkdir(exist_ok=True)
    profile = PASS_PROFILES["phospho"]
    pcfg = _config_for_pass(base, profile)
    assert pcfg.peptide_fdr == 0.05   # must survive the copy


# ---------------------------------------------------------------------------
# 0.5.0: new built-in PTMs, new passes, predicted library + rescoring
# ---------------------------------------------------------------------------

def test_new_builtin_modifications_present():
    """0.5.0 adds O-GlcNAc, Lactyl, Propionyl, Butyryl and Sulfation."""
    for name in ("OGlcNAc", "Lactyl", "Propionyl", "Butyryl", "Sulfation"):
        assert name in DEFAULT_MODIFICATIONS, f"missing built-in mod: {name}"
    assert DEFAULT_MODIFICATIONS["OGlcNAc"].targets == ("S", "T")
    assert DEFAULT_MODIFICATIONS["Lactyl"].targets == ("K",)


def test_new_builtin_passes_present():
    """0.5.0 adds oglcnac, citrullination and lactyl_acyl passes."""
    for name in ("oglcnac", "citrullination", "lactyl_acyl"):
        assert name in PASS_PROFILES
    # oglcnac keeps missed_cleavages at 2 (GlcNAc does not block tryptic cleavage)
    assert PASS_PROFILES["oglcnac"].missed_cleavages == 2
    # lactyl_acyl raises it to 3 like the other K-acyl passes
    assert PASS_PROFILES["lactyl_acyl"].missed_cleavages == 3


def test_map_modifications_expands_residues():
    """PTMQuant mods with multiple target residues become one AlphaPeptDeep
    entry per residue; stem and residue aliases are applied."""
    from diaquant.predicted_library import map_modifications

    mods = [
        DEFAULT_MODIFICATIONS["Carbamidomethyl"],
        DEFAULT_MODIFICATIONS["Phospho"],
        DEFAULT_MODIFICATIONS["GlyGly"],
        DEFAULT_MODIFICATIONS["Acetyl_Nterm"],
    ]
    fix_mods, var_mods = map_modifications(mods)
    assert fix_mods == ["Carbamidomethyl@C"]
    assert "Phospho@S" in var_mods
    assert "Phospho@T" in var_mods
    assert "Phospho@Y" in var_mods
    assert "GG@K" in var_mods
    assert "Acetyl@Protein_N-term" in var_mods


def test_predicted_library_disabled_returns_none(tmp_path):
    """When predicted_library=False the generator short-circuits without
    importing AlphaPeptDeep (works on a stock install)."""
    from diaquant.predicted_library import generate_predicted_library
    fasta = tmp_path / "mini.fasta"
    fasta.write_text(">sp|TEST|TEST_HUMAN\nMAAAAS\n")
    cfg = DiaQuantConfig(
        fasta=fasta,
        mzml_files=[tmp_path / "x.mzML"],
        output_dir=tmp_path,
        predicted_library=False,
    )
    assert generate_predicted_library(cfg, tmp_path) is None


def test_rescore_canonical_key_matches_name_and_mass():
    """Sage Δmass and AlphaPeptDeep name brackets hash to the same key."""
    from diaquant.rescore import _canonical_mod_key
    assert _canonical_mod_key("PEPS[+79.97]TIDE") == \
           _canonical_mod_key("PEPS[Phospho]TIDE")
    assert _canonical_mod_key("ACK[+42.01]LY") == \
           _canonical_mod_key("ACK[Acetyl]LY")


def test_rescore_joins_and_demotes(tmp_path):
    """Matched PSMs get Pred.RT / Pred.RT.Delta; RT outliers beyond the
    tolerance have their Score halved (demoted, not dropped)."""
    from diaquant.rescore import rescore_with_predicted_library

    lib = pd.DataFrame({
        "ModifiedPeptide": ["PEPS[Phospho]TIDE", "ACK[Acetyl]LY"],
        "PrecursorCharge": [2, 2],
        "iRT": [10.2, 25.0],
    })
    lib_path = tmp_path / "lib.tsv"
    lib.to_csv(lib_path, sep="\t", index=False)

    psm = pd.DataFrame({
        "Modified.Sequence": ["PEPS[+79.97]TIDE", "ACK[Acetyl]LY"],
        "Precursor.Charge": [2, 2],
        "RT": [10.0, 20.0],
        "Score": [5.0, 5.0],
        "filename": ["x.mzML", "x.mzML"],
    })
    cfg = DiaQuantConfig(
        fasta=tmp_path / "f.fa",
        mzml_files=[tmp_path / "x.mzML"],
        output_dir=tmp_path,
        rescore_rt_tol_min=3.0,
    )
    out = rescore_with_predicted_library(psm, lib_path, cfg)
    assert abs(out.loc[0, "Pred.RT"] - 10.2) < 1e-6
    assert out.loc[0, "Pred.RT.Delta"] < 1.0
    assert out.loc[0, "Score"] == 5.0
    assert out.loc[1, "Pred.RT.Delta"] > 3.0
    assert out.loc[1, "Score"] == 2.5


def test_rescore_skipped_when_disabled(tmp_path):
    """rescore_with_prediction=False returns the frame unchanged, letting
    users reproduce 0.4.x behaviour exactly."""
    from diaquant.rescore import rescore_with_predicted_library
    psm = pd.DataFrame({
        "Modified.Sequence": ["PEPS[+79.97]TIDE"],
        "Precursor.Charge": [2],
        "RT": [10.0],
        "Score": [5.0],
        "filename": ["x.mzML"],
    })
    cfg = DiaQuantConfig(
        fasta=tmp_path / "f.fa",
        mzml_files=[tmp_path / "x.mzML"],
        output_dir=tmp_path,
        rescore_with_prediction=False,
    )
    out = rescore_with_predicted_library(psm, tmp_path / "missing.tsv", cfg)
    assert list(out.columns) == list(psm.columns)


def test_version_is_050():
    import diaquant
    assert diaquant.__version__ == "0.5.0"
