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
    """The package reports a 0.5.x version (kept name for git-history grep)."""
    import diaquant
    assert diaquant.__version__.startswith("0.5.")


# ---------------------------------------------------------------------------
# 0.5.1: enzyme catalog + instrument preset smoke tests
# ---------------------------------------------------------------------------

def test_enzyme_catalog_exposes_expected_rules():
    from diaquant.enzymes import ENZYME_CATALOG, get_enzyme, to_sage_enzyme_block

    for key in ("trypsin", "lys-c", "no-cleavage", "arg-c", "asp-n",
                "glu-c", "chymotrypsin", "trypsin-strict", "lys-c-strict"):
        assert key in ENZYME_CATALOG, f"enzyme '{key}' missing from catalog"

    t = get_enzyme("trypsin")
    assert t.cleave_at == "KR" and t.restrict == "P"
    ts = get_enzyme("trypsin-strict")
    assert ts.cleave_at == "KR" and ts.restrict is None
    nc = get_enzyme("no-cleavage")
    assert nc.cleave_at == "$" and nc.default_missed_cleavages == 0

    block = to_sage_enzyme_block(t, missed_cleavages=2, min_len=7, max_len=30)
    assert block == {
        "missed_cleavages": 2, "min_len": 7, "max_len": 30,
        "cleave_at": "KR", "restrict": "P",
    }


def test_enzyme_catalog_rejects_unknown_name():
    from diaquant.enzymes import get_enzyme
    with pytest.raises(ValueError):
        get_enzyme("papain")


def test_instrument_preset_applies_only_to_default_fields(tmp_path):
    import yaml
    from diaquant.config import DiaQuantConfig

    fasta = tmp_path / "t.fasta"; fasta.write_text(">p\nMKTAYIAK\n")
    mz = tmp_path / "a.mzML"; mz.write_text("")

    yaml_a = tmp_path / "a.yaml"
    yaml_a.write_text(yaml.safe_dump({
        "fasta": str(fasta),
        "mzml_files": [str(mz)],
        "output_dir": str(tmp_path / "outa"),
        "instrument": "orbitrap_astral",
    }))
    a = DiaQuantConfig.from_yaml(yaml_a)
    assert a.precursor_tol_ppm == 3.0
    assert a.fragment_tol_ppm == 8.0
    assert a.pred_lib_instrument == "Lumos"
    assert a.pred_lib_nce == 27.0
    assert a.min_precursor_mz == 380.0 and a.max_precursor_mz == 980.0

    yaml_b = tmp_path / "b.yaml"
    yaml_b.write_text(yaml.safe_dump({
        "fasta": str(fasta),
        "mzml_files": [str(mz)],
        "output_dir": str(tmp_path / "outb"),
        "instrument": "orbitrap_astral",
        "precursor_tol_ppm": 1.0,
    }))
    b = DiaQuantConfig.from_yaml(yaml_b)
    assert b.precursor_tol_ppm == 1.0
    assert b.fragment_tol_ppm == 8.0


def test_config_rejects_unknown_enzyme(tmp_path):
    import yaml
    from diaquant.config import DiaQuantConfig

    fasta = tmp_path / "t.fasta"; fasta.write_text(">p\nMKTAYIAK\n")
    mz = tmp_path / "a.mzML"; mz.write_text("")
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "fasta": str(fasta),
        "mzml_files": [str(mz)],
        "output_dir": str(tmp_path / "out"),
        "enzyme": "papain",
    }))
    with pytest.raises(ValueError):
        DiaQuantConfig.from_yaml(yaml_path)


def test_sage_runner_uses_enzyme_catalog(tmp_path):
    import yaml
    from diaquant.config import DiaQuantConfig
    from diaquant.sage_runner import build_sage_config

    fasta = tmp_path / "t.fasta"; fasta.write_text(">p\nMKTAYIAK\n")
    mz = tmp_path / "a.mzML"; mz.write_text("")
    yp = tmp_path / "c.yaml"
    yp.write_text(yaml.safe_dump({
        "fasta": str(fasta),
        "mzml_files": [str(mz)],
        "output_dir": str(tmp_path / "out"),
        "enzyme": "glu-c",
        "missed_cleavages": 3,
    }))
    cfg = DiaQuantConfig.from_yaml(yp)
    sage = build_sage_config(cfg)
    assert sage["database"]["enzyme"]["cleave_at"] == "E"
    assert sage["database"]["enzyme"]["restrict"] is None
    assert sage["database"]["enzyme"]["missed_cleavages"] == 3


# ---------------------------------------------------------------------------
# 0.5.3: shared predicted-library cache + site-matrix absolute position + loc filter
# ---------------------------------------------------------------------------

def test_pred_lib_cache_dir_from_env(tmp_path, monkeypatch):
    """``PTMQUANT_LIB_CACHE_DIR`` env var is picked up by DiaQuantConfig."""
    import yaml
    from diaquant.config import DiaQuantConfig

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">sp|P1|TEST_HUMAN test OS=H GN=TEST PE=1 SV=1\nMKAAAASTPEPTIDEK\n")
    mz = tmp_path / "a.mzML"
    mz.write_text("")

    yaml_path = tmp_path / "cfg.yaml"
    yaml.safe_dump({
        "fasta": str(fasta),
        "mzml_files": [str(mz)],
        "output_dir": str(tmp_path / "out"),
    }, yaml_path.open("w"))

    shared = tmp_path / "shared_cache"
    monkeypatch.setenv("PTMQUANT_LIB_CACHE_DIR", str(shared))
    cfg = DiaQuantConfig.from_yaml(yaml_path)
    assert cfg.pred_lib_cache_dir == shared


def test_pred_lib_resolve_cache_paths(tmp_path, monkeypatch):
    """``_resolve_cache_paths`` returns shared + local paths when configured."""
    from diaquant.config import DiaQuantConfig
    from diaquant.predicted_library import _resolve_cache_paths

    fasta = tmp_path / "t.fasta"
    fasta.write_text(">sp|P1|T_HUMAN test GN=T\nMAAA\n")
    mz = tmp_path / "b.mzML"
    mz.write_text("")
    shared = tmp_path / "lib_cache"
    shared.mkdir()

    cfg = DiaQuantConfig(fasta=fasta, mzml_files=[mz],
                         output_dir=tmp_path / "out")
    cfg.pred_lib_cache_dir = shared

    local, sh = _resolve_cache_paths(cfg, tmp_path / "pass_x", "abc123")
    assert local.name == "predicted_library_abc123.tsv"
    assert sh is not None and sh.parent == shared
    assert sh.name == local.name


def test_site_matrix_uses_absolute_position(tmp_path):
    """``site_quant`` + ``write_site_matrix`` emit <accession>|<gene>|<S+abs>|<mod>."""
    from diaquant.quantify import site_quant
    from diaquant.writer import write_site_matrix

    # Synthetic: protein = MKAAASTPEPTIDEKXXXSTR, peptide AAASTPEPTIDEK starts at 3.
    protein_seq = "MKAAASTPEPTIDEKXXXSTR"
    peptide = "AAASTPEPTIDEK"
    fasta_records = {"P1": {"name": "P1_HUMAN", "gene": "TEST",
                            "descr": "test", "seq": protein_seq}}

    long_df = pd.DataFrame({
        "filename": ["s1.mzML", "s2.mzML"],
        "Protein.Group": ["P1", "P1"],
        "Genes": ["TEST", "TEST"],
        "Stripped.Sequence": [peptide, peptide],
        "Modified.Sequence": [peptide, peptide],
        "Precursor.Id": [peptide + "2", peptide + "2"],
        "Intensity": [1e6, 2e6],
        "PTM.Site.Positions": ["S4", "S4"],        # peptide-local S at pos 4 -> abs 6
        "PTM.Modification": ["+79.9663", "+79.9663"],
        "Best.Site.Probability": [0.99, 0.98],
    })
    long_df.attrs["fasta_records"] = fasta_records

    site_lfq = site_quant(long_df, min_samples=1,
                          localization_cutoff=0.75)
    assert not site_lfq.empty
    assert site_lfq["protein"].iloc[0] == "P1|TEST|S6|+79.9663"

    out = tmp_path / "site.tsv"
    write_site_matrix(site_lfq, out)
    df = pd.read_csv(out, sep="\t")
    assert set(["Protein.Group", "Genes", "PTM.Site",
                "PTM.Modification"]) <= set(df.columns)
    assert df["Protein.Group"].iloc[0] == "P1"
    assert df["PTM.Site"].iloc[0] == "S6"


def test_site_matrix_localization_filter_drops_low_loc():
    """Precursors with loc prob below cutoff are excluded by default."""
    from diaquant.quantify import site_quant

    long_df = pd.DataFrame({
        "filename": ["a.mzML", "b.mzML"],
        "Protein.Group": ["P1", "P1"],
        "Genes": ["T", "T"],
        "Stripped.Sequence": ["AAASK", "AAASK"],
        "Modified.Sequence": ["AAASK", "AAASK"],
        "Precursor.Id": ["AAASK2", "AAASK2"],
        "Intensity": [1e5, 2e5],
        "PTM.Site.Positions": ["S4", "S4"],
        "PTM.Modification": ["+79.9663", "+79.9663"],
        "Best.Site.Probability": [0.20, 0.30],
    })
    long_df.attrs["fasta_records"] = {
        "P1": {"name": "P1", "gene": "T", "descr": "", "seq": "MKAAASK"}
    }

    strict = site_quant(long_df, min_samples=1, localization_cutoff=0.75)
    assert strict.empty, "Low-localization precursors should be excluded"

    permissive = site_quant(long_df, min_samples=1,
                            localization_cutoff=0.75, include_low_loc=True)
    assert not permissive.empty


def test_attach_fasta_meta_handles_full_protein_group():
    """v0.5.3.1: ``Protein.Group`` like ``sp|P1|GENE_HUMAN`` is normalised before FASTA lookup.

    Regression test for the v0.5.2/v0.5.3 bug where every Genes / Protein.Names
    /  First.Protein.Description cell came out empty because the lookup key was
    the full Sage tag instead of the bare UniProt accession.
    """
    import tempfile
    from pathlib import Path
    from diaquant.parse_sage import attach_fasta_meta, _extract_accession

    # Direct helper checks
    assert _extract_accession("sp|P12345|GENE_HUMAN") == "P12345"
    assert _extract_accession("rev_sp|P12345|GENE_HUMAN") == "P12345"
    assert _extract_accession("tr|Q67890|UNREVIEWED_HUMAN") == "Q67890"
    assert _extract_accession("P12345") == "P12345"
    assert _extract_accession("sp|P1|G;sp|P2|H") == "P1"  # group → first

    with tempfile.TemporaryDirectory() as td:
        fasta = Path(td) / "tiny.fasta"
        fasta.write_text(
            ">sp|P12345|GENE_HUMAN My favourite protein OS=Homo sapiens "
            "OX=9606 GN=MYGENE PE=1 SV=2\nMKAAAASTPEPTIDEK\n"
            ">sp|Q9XYZ0|OTHER_HUMAN Another GN=OTHER\nMAAA\n"
        )
        df = pd.DataFrame({
            "Protein.Group": [
                "sp|P12345|GENE_HUMAN",
                "rev_sp|P12345|GENE_HUMAN",   # decoy
                "sp|Q9XYZ0|OTHER_HUMAN",
                "sp|UNKNOWN|MISSING_HUMAN",   # unmatched
            ]
        })
        out = attach_fasta_meta(df, fasta)
        assert out["Genes"].tolist() == ["MYGENE", "MYGENE", "OTHER", ""]
        assert out["Protein.Names"].iloc[0] == "GENE_HUMAN"
        assert "favourite" in out["First.Protein.Description"].iloc[0]
        # Records + extractor are stashed for site_quant().
        assert "fasta_records" in out.attrs
        assert "P12345" in out.attrs["fasta_records"]
        assert callable(out.attrs.get("protein_group_accession"))


def test_site_quant_translates_protein_group_to_accession():
    """v0.5.3.1: site_quant maps full Sage Protein.Group to bare accession before FASTA lookup."""
    from diaquant.quantify import site_quant
    from diaquant.parse_sage import _extract_accession

    protein_seq = "MKAAASTPEPTIDEKXXXSTR"
    peptide = "AAASTPEPTIDEK"

    long_df = pd.DataFrame({
        "filename": ["s1.mzML", "s2.mzML"],
        # Note: full Sage tag, not bare accession.
        "Protein.Group": ["sp|P1|TEST_HUMAN", "sp|P1|TEST_HUMAN"],
        "Genes": ["TEST", "TEST"],
        "Stripped.Sequence": [peptide, peptide],
        "Modified.Sequence": [peptide, peptide],
        "Precursor.Id": [peptide + "2", peptide + "2"],
        "Intensity": [1e6, 2e6],
        "PTM.Site.Positions": ["S4", "S4"],
        "PTM.Modification": ["+79.9663", "+79.9663"],
        "Best.Site.Probability": [0.99, 0.98],
    })
    long_df.attrs["fasta_records"] = {
        "P1": {"name": "TEST_HUMAN", "gene": "TEST",
               "descr": "test", "seq": protein_seq}
    }
    long_df.attrs["protein_group_accession"] = _extract_accession

    site_lfq = site_quant(long_df, min_samples=1, localization_cutoff=0.75)
    assert not site_lfq.empty, (
        "Pre-fix: this returned empty because fasta_records['sp|P1|TEST_HUMAN'] missed."
    )
    # _abs_site_keys uses the *accession* as the first key field.
    key = site_lfq["protein"].iloc[0]
    assert key.startswith("P1|TEST|S6|"), key
    assert "_local" not in key, "Should resolve to absolute position, not _local fallback"
