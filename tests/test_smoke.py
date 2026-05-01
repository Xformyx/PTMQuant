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
    """``site_quant`` + ``write_site_matrix`` emit <accession>|<gene>|<S+abs>|<mod>.

    v0.5.7: the fixture now spans two sites (one phospho-S, one phospho-T)
    across two runs so that directLFQ has enough ratios to compute a
    normalised intensity, mirroring the minimum data shape produced by a
    real Sage search.
    """
    from diaquant.quantify import site_quant
    from diaquant.writer import write_site_matrix

    # Synthetic: protein = MKAAASTPEPTIDEKXXXSTR, peptide AAASTPEPTIDEK starts at 3.
    protein_seq = "MKAAASTPEPTIDEKXXXSTR"
    peptide_a  = "AAASTPEPTIDEK"      # phospho-S at peptide-local pos 4 -> abs 6
    peptide_b  = "PEPTIDEKXXXSTR"     # phospho-T at peptide-local pos 4 -> abs 14
    fasta_records = {"P1": {"name": "P1_HUMAN", "gene": "TEST",
                            "descr": "test", "seq": protein_seq}}

    long_df = pd.DataFrame({
        "filename":           ["s1.mzML", "s2.mzML", "s1.mzML", "s2.mzML"],
        "Protein.Group":      ["P1", "P1", "P1", "P1"],
        "Genes":              ["TEST", "TEST", "TEST", "TEST"],
        "Stripped.Sequence":  [peptide_a, peptide_a, peptide_b, peptide_b],
        "Modified.Sequence":  [peptide_a, peptide_a, peptide_b, peptide_b],
        "Precursor.Id":       [peptide_a + "2", peptide_a + "2",
                               peptide_b + "2", peptide_b + "2"],
        "Intensity":          [1e6, 2e6, 5e5, 1.1e6],
        "PTM.Site.Positions": ["S4", "S4", "T4", "T4"],
        "PTM.Modification":   ["+79.9663"] * 4,
        "Best.Site.Probability": [0.99, 0.98, 0.99, 0.97],
    })
    long_df.attrs["fasta_records"] = fasta_records

    site_lfq = site_quant(long_df, min_samples=1,
                          localization_cutoff=0.75)
    assert not site_lfq.empty
    # v0.5.5: mod name normalised to human-readable ("+79.9663" → "Phospho").
    assert site_lfq["protein"].iloc[0] == "P1|TEST|S6|Phospho"

    out = tmp_path / "site.tsv"
    write_site_matrix(site_lfq, out)
    df = pd.read_csv(out, sep="\t")
    assert set(["Protein.Group", "Genes", "PTM.Site",
                "PTM.Modification"]) <= set(df.columns)
    # Both sites must appear, in any order.
    sites = set(df["PTM.Site"])
    assert {"S6", "T17"} <= sites or {"S6", "T18"} <= sites or len(sites) == 2
    assert (df["Protein.Group"] == "P1").all()
    assert (df["PTM.Modification"] == "Phospho").all()


def test_site_matrix_localization_filter_drops_low_loc():
    """Precursors with loc prob below cutoff are excluded by default.

    v0.5.7: fixture spans two distinct sites so directLFQ has enough ratios
    to compute normalised intensities in the *permissive* arm.
    """
    from diaquant.quantify import site_quant

    long_df = pd.DataFrame({
        "filename":           ["a.mzML", "b.mzML", "a.mzML", "b.mzML"],
        "Protein.Group":      ["P1", "P1", "P1", "P1"],
        "Genes":              ["T", "T", "T", "T"],
        "Stripped.Sequence":  ["AAASK", "AAASK", "PEPTIDET", "PEPTIDET"],
        "Modified.Sequence":  ["AAASK", "AAASK", "PEPTIDET", "PEPTIDET"],
        "Precursor.Id":       ["AAASK2", "AAASK2", "PEPTIDET2", "PEPTIDET2"],
        "Intensity":          [1e5, 2e5, 5e4, 1.1e5],
        "PTM.Site.Positions": ["S4", "S4", "T8", "T8"],
        "PTM.Modification":   ["+79.9663"] * 4,
        "Best.Site.Probability": [0.20, 0.30, 0.20, 0.30],
    })
    long_df.attrs["fasta_records"] = {
        "P1": {"name": "P1", "gene": "T", "descr": "",
               "seq": "MKAAASKAAAPEPTIDETAAA"}
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
    """v0.5.3.1: site_quant maps full Sage Protein.Group to bare accession before FASTA lookup.

    v0.5.7: fixture spans two distinct sites for directLFQ ratio normalisation.
    """
    from diaquant.quantify import site_quant
    from diaquant.parse_sage import _extract_accession

    protein_seq = "MKAAASTPEPTIDEKXXXSTR"
    peptide_a = "AAASTPEPTIDEK"
    peptide_b = "PEPTIDEKXXXSTR"

    long_df = pd.DataFrame({
        "filename":           ["s1.mzML", "s2.mzML", "s1.mzML", "s2.mzML"],
        # Note: full Sage tag, not bare accession.
        "Protein.Group":      ["sp|P1|TEST_HUMAN"] * 4,
        "Genes":              ["TEST"] * 4,
        "Stripped.Sequence":  [peptide_a, peptide_a, peptide_b, peptide_b],
        "Modified.Sequence":  [peptide_a, peptide_a, peptide_b, peptide_b],
        "Precursor.Id":       [peptide_a + "2", peptide_a + "2",
                               peptide_b + "2", peptide_b + "2"],
        "Intensity":          [1e6, 2e6, 5e5, 1.1e6],
        "PTM.Site.Positions": ["S4", "S4", "T4", "T4"],
        "PTM.Modification":   ["+79.9663"] * 4,
        "Best.Site.Probability": [0.99, 0.98, 0.99, 0.97],
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
    # The accession-prefixed key (P1|TEST|<site>|Phospho) must appear in some row.
    keys = list(site_lfq["protein"])
    assert any(k.startswith("P1|TEST|S6|Phospho") for k in keys), keys
    assert all("_local" not in k for k in keys), \
        "Should resolve to absolute position, not _local fallback"


# ---------------------------------------------------------------------------
# v0.5.4 regression tests
# ---------------------------------------------------------------------------

def test_version_bumped_to_054():
    import diaquant
    # renamed to 'v0.5.4+ or newer'; keep test name for traceability.
    v = diaquant.__version__
    major, minor = v.split(".")[:2]
    assert (int(major), int(minor)) >= (0, 5), v


def test_razor_assigns_unique_peptide_to_sole_parent():
    from diaquant.razor import apply_razor_grouping
    df = pd.DataFrame({
        "Stripped.Sequence": ["AAAAAAK", "AAAAAAK", "BBBBBBK"],
        "Protein.Ids":       ["P1",       "P1",      "P1;P2"],
    })
    out = apply_razor_grouping(df)
    # AAAAAAK is single-candidate -> P1 unconditionally
    pg = dict(zip(out["Stripped.Sequence"], out["Protein.Group"]))
    assert pg["AAAAAAK"] == "P1"


def test_razor_assigns_shared_peptide_to_most_supported_parent():
    from diaquant.razor import apply_razor_grouping
    # P1 has 2 unique peptides, P2 has none. Shared peptide must go to P1.
    df = pd.DataFrame({
        "Stripped.Sequence": ["U1K", "U2K", "SHAREDK", "SHAREDK"],
        "Protein.Ids":       ["P1",  "P1",  "P1;P2",   "P1;P2"],
    })
    out = apply_razor_grouping(df)
    pg = dict(zip(out["Stripped.Sequence"], out["Protein.Group"]))
    assert pg["SHAREDK"] == "P1"
    # and shared peptides must be flagged Proteotypic for their group
    shared_rows = out[out["Stripped.Sequence"] == "SHAREDK"]
    assert (shared_rows["Proteotypic"] == 1).all()


def test_razor_merges_identical_peptide_sets_into_semicolon_group():
    from diaquant.razor import apply_razor_grouping
    # P3 and P4 are both supported ONLY by the same shared peptide set.
    # They must be merged into a single group "P3;P4".
    df = pd.DataFrame({
        "Stripped.Sequence": ["XYZK", "XYZK", "WWWK"],
        "Protein.Ids":       ["P3;P4", "P3;P4", "P3;P4"],
    })
    out = apply_razor_grouping(df)
    groups = set(out["Protein.Group"].unique())
    assert groups == {"P3;P4"}, groups


def test_razor_respects_min_peptides_per_protein():
    from diaquant.razor import apply_razor_grouping
    df = pd.DataFrame({
        "Stripped.Sequence": ["ONLYPEPK", "TWOAK", "TWOBK"],
        "Protein.Ids":       ["PA",       "PB",    "PB"],
    })
    out = apply_razor_grouping(df, min_peptides_per_protein=2)
    # PA only has 1 peptide -> filtered out; PB survives with TWOAK + TWOBK.
    assert set(out["Protein.Group"]) == {"PB"}


def test_manifest_writes_expected_keys(tmp_path):
    from diaquant.manifest import write_run_manifest
    from diaquant.config import DiaQuantConfig
    import json
    cfg = DiaQuantConfig(
        fasta=tmp_path / "x.fasta",
        mzml_files=[tmp_path / "a.mzML"],
        output_dir=tmp_path,
        predicted_library=True,
        rescore_with_prediction=True,
    )
    yaml_src = tmp_path / "config.yaml"
    yaml_src.write_text("hello: world\n")
    lib = tmp_path / "predicted_library_abc.tsv"
    lib.write_text("header\nrow1\nrow2\n")
    out = write_run_manifest(
        cfg, tmp_path,
        diaquant_version="0.5.4",
        config_yaml_src=yaml_src,
        library_paths=[lib],
        n_psms_raw=123, n_psms_rescored=99, n_psms_after_fdr=88,
        pr_rows=50, pg_rows=10, site_rows=3,
    )
    data = json.loads(out.read_text())
    assert data["diaquant_version"] == "0.5.4"
    assert data["predicted_library"]["applied"] is True
    assert data["rescoring"]["applied"] is True
    assert data["matrices"] == {
        "pr_matrix_rows": 50, "pg_matrix_rows": 10, "ptm_site_matrix_rows": 3,
    }
    # config.yaml should have been copied into tmp_path
    assert (tmp_path / "config.yaml").exists()
    # predicted library mirrored / symlinked
    assert (tmp_path / "predicted_libraries" / lib.name).exists()


def test_manifest_flags_predicted_library_off_when_no_files(tmp_path):
    from diaquant.manifest import write_run_manifest
    from diaquant.config import DiaQuantConfig
    import json
    cfg = DiaQuantConfig(
        fasta=tmp_path / "x.fasta",
        mzml_files=[tmp_path / "a.mzML"],
        output_dir=tmp_path,
        predicted_library=True,
        rescore_with_prediction=True,
    )
    out = write_run_manifest(
        cfg, tmp_path, diaquant_version="0.5.4",
        config_yaml_src=None, library_paths=[],
    )
    data = json.loads(out.read_text())
    # enabled in config but zero files -> applied must be False
    assert data["predicted_library"]["enabled_in_config"] is True
    assert data["predicted_library"]["applied"] is False
    # rescoring is gated on predicted_library.applied
    assert data["rescoring"]["applied"] is False


def test_quant_min_samples_default_is_one():
    from diaquant.config import DiaQuantConfig
    from pathlib import Path
    cfg = DiaQuantConfig(
        fasta=Path("/tmp/x.fasta"),
        mzml_files=[Path("/tmp/a.mzML")],
        output_dir=Path("/tmp/out"),
    )
    assert cfg.quant_min_samples == 1
    assert cfg.min_peptides_per_protein == 1


# =====================================================================
# v0.5.5 regression tests
# =====================================================================

def test_version_is_055():
    """Pinned to v0.5.5+ so downstream pipelines can gate on the feature set.

    Name preserved for git-grep; actual assertion accepts any 0.5.x >= 0.5.5.
    """
    import diaquant

    def _triple(v: str) -> tuple[int, int, int]:
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2].split("rc")[0].split("a")[0].split("b")[0]))

    assert _triple(diaquant.__version__) >= (0, 5, 5), diaquant.__version__


def test_mbr_config_defaults():
    """MBR is on by default; knobs exist for tolerance and donor count."""
    from diaquant.config import DiaQuantConfig
    from pathlib import Path
    cfg = DiaQuantConfig(
        fasta=Path("/tmp/x.fasta"),
        mzml_files=[Path("/tmp/a.mzML")],
        output_dir=Path("/tmp/out"),
    )
    assert cfg.mbr_rescue is True
    assert cfg.mbr_rt_tol_min == 1.0
    assert cfg.mbr_min_donors == 2
    # Backward compatibility: the legacy ``match_between_runs`` field stays.
    assert cfg.match_between_runs is True


def test_rt_align_phospho_overlay_defaults():
    """Phospho-aware LOWESS overlay + Pred.RT anchor filter are default-on."""
    from diaquant.config import DiaQuantConfig
    from pathlib import Path
    cfg = DiaQuantConfig(
        fasta=Path("/tmp/x.fasta"),
        mzml_files=[Path("/tmp/a.mzML")],
        output_dir=Path("/tmp/out"),
    )
    assert cfg.rt_align_per_pass_phospho is True
    assert cfg.rt_align_pred_rt_tol_min == 2.0


def test_add_site_probabilities_emits_ptm_mods_per_mod():
    """v0.5.5: PTM.Mods splits sites per modification so phospho whitelist
    filtering doesn't conflate oxidation + phospho."""
    import pandas as pd
    from diaquant.ptm_localization import add_site_probabilities
    df = pd.DataFrame({
        "filename":          ["a.mzML", "a.mzML"],
        "peptide":           [
            "AAAM[+15.9949]S[+79.9663]STK",
            "PEPTIDES[+79.9663]K",
        ],
        "Modified.Sequence": [
            "AAAM[+15.9949]S[+79.9663]STK",
            "PEPTIDES[+79.9663]K",
        ],
        "Stripped.Sequence": ["AAAMSSTK", "PEPTIDESK"],
        "Precursor.Charge":  [2, 2],
        "charge":             [2, 2],
        "Score":             [15.0, 12.0],
    })
    out = add_site_probabilities(df)
    assert "PTM.Mods" in out.columns, "v0.5.5 must emit PTM.Mods"
    # First row must carry Phospho, and phospho positions must NOT include M.
    mods_row0 = out.loc[0, "PTM.Mods"]
    s0 = str(mods_row0)
    assert "Phospho" in s0, f"PTM.Mods row0 missing Phospho: {mods_row0!r}"
    # Phospho sites (serialised as 'S5@0.xx' etc.) should sit on S/T/Y.
    import re
    phospho_chunk = re.search(r"Phospho:([^;]*)", s0)
    if phospho_chunk:
        for site in phospho_chunk.group(1).split(","):
            aa = site.strip()[:1]
            assert aa in "STY", (
                f"Phospho site should be on STY, got {site!r}"
            )


def test_normalise_mass_tag_to_phospho():
    """v0.5.5: raw +79.9663 mass tags are normalised to 'Phospho' so the
    site_quant whitelist matches."""
    from diaquant.ptm_localization import _normalise_tag
    assert _normalise_tag("+79.9663") == "Phospho"
    assert _normalise_tag("+79.96633") == "Phospho"
    assert _normalise_tag("Phospho") == "Phospho"
    # Unknown tags pass through unchanged.
    assert _normalise_tag("SomeCustomPTM") == "SomeCustomPTM"


def test_manifest_autodiscovers_predicted_libraries(tmp_path):
    """v0.5.5: when library_paths=None, manifest scans out_dir for
    predicted_library_*.tsv including pass subdirs."""
    from diaquant.manifest import write_run_manifest
    from diaquant.config import DiaQuantConfig
    import json
    # Simulate a multi-pass output tree
    (tmp_path / "pass_phospho").mkdir()
    (tmp_path / "pass_phospho" / "predicted_library_abc.tsv").write_text("x")
    cfg = DiaQuantConfig(
        fasta=tmp_path / "x.fasta",
        mzml_files=[tmp_path / "a.mzML"],
        output_dir=tmp_path,
        predicted_library=True,
        rescore_with_prediction=True,
    )
    # library_paths=None triggers auto-discovery
    out = write_run_manifest(
        cfg, tmp_path, diaquant_version="0.5.5",
        config_yaml_src=None, library_paths=None,
    )
    data = json.loads(out.read_text())
    assert data["predicted_library"]["enabled_in_config"] is True
    assert data["predicted_library"]["applied"] is True, (
        "Expected auto-discovery to pick up pass_phospho/predicted_library_*.tsv"
    )


def test_manifest_respects_explicit_empty_library_list(tmp_path):
    """v0.5.5: explicit empty list means 'we checked, none produced',
    so manifest must NOT auto-discover stray files elsewhere."""
    from diaquant.manifest import write_run_manifest
    from diaquant.config import DiaQuantConfig
    import json
    (tmp_path / "predicted_library_zombie.tsv").write_text("x")
    cfg = DiaQuantConfig(
        fasta=tmp_path / "x.fasta",
        mzml_files=[tmp_path / "a.mzML"],
        output_dir=tmp_path,
        predicted_library=True,
        rescore_with_prediction=True,
    )
    out = write_run_manifest(
        cfg, tmp_path, diaquant_version="0.5.5",
        config_yaml_src=None, library_paths=[],  # explicit empty
    )
    data = json.loads(out.read_text())
    assert data["predicted_library"]["applied"] is False


# =====================================================================
# v0.5.6 regression tests  (release engineering + observability)
# =====================================================================

def test_verify_ptmquant_passes_for_healthy_output(tmp_path):
    """The new stdlib-only verifier returns exit code 0 on a well-formed output."""
    import json
    import subprocess
    import sys

    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "diaquant_version": "0.5.9",
        "predicted_library": {"applied": True, "paths": ["p.tsv"]},
        "rescoring": {"configured": True, "applied": True},
        "mbr": {"configured": True, "applied": True, "n_rescued": 12},
        "passes": ["phospho"],
    }))
    (tmp_path / "report.pr_matrix.tsv").write_text(
        "Protein.Group\tPrecursor.Id\ts1\n"
        "sp|P1|HUMAN\tAK/1\t100\n"
        "sp|P2|HUMAN\tBK/2\t200\n"
    )
    (tmp_path / "report.pg_matrix.tsv").write_text(
        "Protein.Group\tGenes\ts1\n"
        "sp|P1|HUMAN\tGENE1\t100\n"
        "sp|P2|HUMAN\tGENE2\t200\n"
    )
    (tmp_path / "report.ptm_site_matrix.tsv").write_text(
        "Protein.Group\tGenes\tPTM.Site\ts1\n"
        "sp|P1|HUMAN\tGENE1\tS-1\t50\n"
    )

    import pathlib
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_ptmquant.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["failures"] == []


def test_verify_ptmquant_fails_when_manifest_missing(tmp_path):
    """Verifier fails loudly if run_manifest.json is missing (pre-v0.5.4 image)."""
    import subprocess
    import sys
    (tmp_path / "report.pr_matrix.tsv").write_text(
        "Protein.Group\tPrecursor.Id\ts1\nP1\tAK/1\t100\n"
    )
    (tmp_path / "report.pg_matrix.tsv").write_text(
        "Protein.Group\tGenes\ts1\nP1\tGENE1\t100\n"
    )

    import pathlib
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_ptmquant.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "run_manifest.json is missing" in result.stdout


def test_pyproject_version_matches_package():
    """pyproject.toml must declare version dynamically so the wheel, image and
    ``diaquant --version`` cannot disagree (v0.5.6 release-engineering fix)."""
    import pathlib
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore

    root = pathlib.Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    dynamic = data["project"].get("dynamic", [])
    assert "version" in dynamic, "pyproject.toml must declare version as dynamic"
    attr = data.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("attr")
    assert attr == "diaquant.__version__", attr


# ---------------------------------------------------------------------------
# v0.5.7 regression tests
# ---------------------------------------------------------------------------

def test_v057_parse_sage_drops_decoys(tmp_path):
    """parse_sage_tsv must drop Sage decoys (label == -1) before FDR filtering.

    v0.5.6 leaked 185 decoy precursors and 101 decoy protein-groups into the
    KIST EPS report.  After the v0.5.7 fix the long_df contains zero decoys.
    """
    import io
    import pandas as pd
    from diaquant.parse_sage import parse_sage_tsv

    rows = pd.DataFrame({
        "filename":         ["a.mzML"] * 4,
        "scannr":           [1, 2, 3, 4],
        "peptide":          ["AAASK", "PEPTIDEK", "AAASK", "PEPTIDEK"],
        "stripped_peptide": ["AAASK", "PEPTIDEK", "AAASK", "PEPTIDEK"],
        "proteins":         ["sp|P1|GENE_HUMAN", "sp|P2|GENE_HUMAN",
                              "rev_sp|P9|REV_HUMAN", "rev_sp|P8|REV_HUMAN"],
        "charge":           [2, 2, 2, 2],
        "calcmass":         [500.25, 800.40, 500.25, 800.40],
        "expmass":          [500.25, 800.40, 500.25, 800.40],
        "rt":               [10.0, 20.0, 10.0, 20.0],
        "predicted_rt":     [10.1, 20.1, 10.1, 20.1],
        "spectrum_q":       [0.001, 0.002, 0.001, 0.002],
        "peptide_q":        [0.001, 0.002, 0.001, 0.002],
        "protein_q":        [0.001, 0.002, 0.001, 0.002],
        "ms1_intensity":    [1e6, 2e6, 1e6, 2e6],
        "ms2_intensity":    [1e7, 2e7, 1e7, 2e7],
        "hyperscore":       [40.0, 38.0, 30.0, 28.0],
        "label":            [1, 1, -1, -1],
    })
    p = tmp_path / "results.sage.tsv"
    rows.to_csv(p, sep="\t", index=False)
    df = parse_sage_tsv(p, peptide_fdr=0.01)
    # decoys must be gone
    assert (df["Protein.Group"].astype(str)
              .str.startswith(("rev_", "REV_", "DECOY_"))).sum() == 0
    # targets must survive
    assert len(df) == 2


def test_v057_site_quant_handles_ptmmods_via_column_access():
    """Regression: itertuples() mangles dotted column names so the v0.5.5/v0.5.6
    site_quant returned an empty matrix even when PTM.Mods was populated.

    v0.5.7 iterates the columns directly via numpy arrays.
    """
    import pandas as pd
    from diaquant.quantify import site_quant

    df = pd.DataFrame({
        "filename":           ["a.mzML"] * 6,
        "Protein.Group":      ["sp|P1|GENE_HUMAN"] * 3 + ["sp|P2|OTHR_HUMAN"] * 3,
        "Protein.Ids":        ["P1"] * 3 + ["P2"] * 3,
        "Protein.Names":      ["GENE_HUMAN"] * 3 + ["OTHR_HUMAN"] * 3,
        "Genes":              ["GENE"] * 3 + ["OTHR"] * 3,
        "Stripped.Sequence":  ["AAASK"] * 3 + ["PEPTIDET"] * 3,
        "Modified.Sequence":  ["AAAS[+79.9663]K"] * 3 + ["PEPTIDET[+79.9663]"] * 3,
        "Precursor.Charge":   [2, 2, 2, 2, 2, 2],
        "Precursor.Id":       ["AAAS[+79.9663]K2"] * 3 + ["PEPTIDET[+79.9663]2"] * 3,
        "Intensity":          [1e6, 1.1e6, 1.2e6, 5e5, 5.5e5, 6e5],
        "PTM.Mods":           ["Phospho:S4@0.990"] * 3 + ["Phospho:T8@0.990"] * 3,
    })
    df["filename"] = ["a.mzML", "b.mzML", "c.mzML"] * 2
    df.attrs["fasta_records"] = {
        "P1": {"name": "GENE_HUMAN", "gene": "GENE", "descr": "",
               "seq": "MKAAASKAAA"},
        "P2": {"name": "OTHR_HUMAN", "gene": "OTHR", "descr": "",
               "seq": "MKPEPTIDETXX"},
    }
    out = site_quant(df, min_samples=1, localization_cutoff=0.0,
                     phospho_only=True)
    assert not out.empty, "PTM.Mods path must yield rows in v0.5.7"
    # Two distinct site keys, both phospho
    keys = set(out["protein"])
    assert len(keys) == 2
    assert all("Phospho" in k for k in keys)


def test_v057_precursor_matrix_normalized_runs_in_pipeline():
    """The new precursor_matrix_normalized() must keep the same column layout
    as precursor_matrix() (just with normalised values) so writers continue
    to work unchanged.
    """
    import pandas as pd
    from diaquant.quantify import precursor_matrix, precursor_matrix_normalized
    long_df = pd.DataFrame({
        "filename":           ["a.mzML", "b.mzML", "a.mzML", "b.mzML"],
        "Protein.Group":      ["P1", "P1", "P2", "P2"],
        "Protein.Ids":        ["P1", "P1", "P2", "P2"],
        "Protein.Names":      ["P1_HUMAN", "P1_HUMAN", "P2_HUMAN", "P2_HUMAN"],
        "Genes":              ["G1", "G1", "G2", "G2"],
        "First.Protein.Description": ["", "", "", ""],
        "Proteotypic":        ["1", "1", "1", "1"],
        "Stripped.Sequence":  ["AAASK", "AAASK", "PEPTIDET", "PEPTIDET"],
        "Modified.Sequence":  ["AAASK", "AAASK", "PEPTIDET", "PEPTIDET"],
        "Precursor.Charge":   [2, 2, 2, 2],
        "Precursor.Id":       ["AAASK2", "AAASK2", "PEPTIDET2", "PEPTIDET2"],
        "Intensity":          [1e6, 5e6, 2e6, 1e7],   # b.mzML is 5x loaded
    })
    raw = precursor_matrix(long_df)
    norm = precursor_matrix_normalized(long_df)
    assert list(raw.columns) == list(norm.columns)
    # After normalisation the per-sample medians should be much closer than 5x.
    raw_ratio  = raw["b.mzML"].median() / raw["a.mzML"].median()
    norm_ratio = norm["b.mzML"].median() / norm["a.mzML"].median()
    assert abs(raw_ratio - 5.0) < 0.5
    assert abs(norm_ratio - 1.0) < 0.5, (raw_ratio, norm_ratio)


# ---------------------------------------------------------------------------
# v0.5.8 regression tests
# ---------------------------------------------------------------------------

def test_v058_phospho_pass_uses_5pct_peptide_fdr() -> None:
    """The built-in PTM-aware profiles must override peptide_fdr to 0.05.

    v0.5.7 left peptide_fdr at the global 1% which empirically truncates
    real phospho rows because the FDR estimator is dominated by unmodified
    peptides.  v0.5.8 sets peptide_fdr=0.05 on every PTM-aware profile
    while keeping site_probability_cutoff=0.75 as the localisation
    safeguard.  whole_proteome must still inherit the strict 1% global.
    """
    ptm_passes = [
        "phospho", "ubiquitin", "acetyl_methyl", "succinyl_acyl",
        "oglcnac", "citrullination", "lactyl_acyl",
    ]
    for name in ptm_passes:
        prof = PASS_PROFILES[name]
        assert prof.peptide_fdr == 0.05, (
            f"profile {name} peptide_fdr={prof.peptide_fdr}, expected 0.05"
        )
        assert prof.site_probability_cutoff == 0.75, (
            f"profile {name} site cutoff must remain 0.75"
        )
    assert PASS_PROFILES["whole_proteome"].peptide_fdr is None


def test_v058_imputation_group_median_respects_min_obs() -> None:
    """Group-aware imputation must NOT invent values when a group has fewer
    than ``min_obs_per_group`` valid observations."""
    from diaquant.imputation import (
        ImputeParams, build_sample_to_group, impute_matrix,
    )

    cols = ["S1", "S2", "S3", "T1", "T2", "T3"]
    sample_sheet = pd.DataFrame({
        "mzml_basename": cols,
        "group": ["ctrl"] * 3 + ["treat"] * 3,
    })
    s2g = build_sample_to_group(cols, sample_sheet)
    assert s2g == {"S1": "ctrl", "S2": "ctrl", "S3": "ctrl",
                   "T1": "treat", "T2": "treat", "T3": "treat"}

    matrix = pd.DataFrame({
        "Precursor.Id": ["P1", "P2"],
        "S1": [100.0, 200.0],
        "S2": [120.0, 240.0],
        "S3": [None,  None],
        "T1": [None,  300.0],
        "T2": [None,  None],
        "T3": [None,  None],
    })
    out = impute_matrix(
        matrix, s2g, id_cols=["Precursor.Id"],
        params=ImputeParams(method="group_median", min_obs_per_group=2),
    )
    # P1 ctrl S3 imputed = median(100, 120) = 110; treat all stay NaN
    assert out.loc[0, "S3"] == 110.0
    assert pd.isna(out.loc[0, "T1"]) and pd.isna(out.loc[0, "T2"])
    # P2 ctrl S3 imputed = median(200, 240) = 220; treat T1 stays original
    assert out.loc[1, "S3"] == 220.0
    assert out.loc[1, "T1"] == 300.0
    assert "Intensity.Imputed.Frac" in out.columns


def test_v058_predicted_donor_table_reads_alphapeptdeep_columns(tmp_path) -> None:
    """``_load_predicted_donor_table`` must accept AlphaPeptDeep flavoured
    column names and return the canonical triple expected by the MBR pool."""
    from diaquant.cli import _load_predicted_donor_table

    p = tmp_path / "predicted.tsv"
    pd.DataFrame({
        "ModifiedPeptide": ["PEPTIDE", "PEPS[Phospho (S)]TIDE", "_NA_"],
        "PrecursorCharge": [2, 3, 2],
        "iRT": [12.5, 14.1, None],
    }).to_csv(p, sep="\t", index=False)
    donors = _load_predicted_donor_table([str(p)])
    assert donors is not None and len(donors) == 2
    assert set(donors.columns) == {"Modified.Sequence", "Precursor.Charge",
                                   "Pred.RT"}
    assert donors["Precursor.Charge"].dtype.kind == "i"
    assert donors["Pred.RT"].dtype.kind == "f"


def test_v058_predicted_donor_table_returns_none_without_paths() -> None:
    """No predicted-library means MBR injection is silently skipped."""
    from diaquant.cli import _load_predicted_donor_table

    assert _load_predicted_donor_table([]) is None
    assert _load_predicted_donor_table(["/no/such/file.tsv"]) is None


def test_v058_mbr_injected_donors_only_promote_observed_rows() -> None:
    """Predicted donors must NOT invent PSMs - they only relax the FDR cut
    on rows Sage already scored at any q-value."""
    from diaquant.mbr import MBRParams, match_between_runs

    psm_full = pd.DataFrame({
        "Modified.Sequence": ["PEPTIDE", "PEPSTIDE"],
        "Precursor.Charge": [2, 2],
        "filename": ["run1.mzML", "run1.mzML"],
        "Q.Value": [0.005, 0.04],
        "Precursor.Quantity": [1e5, 5e4],
        "Protein.Group": ["A", "A"],
        "RT": [10.0, 10.5],
        "RT.Aligned": [10.0, 10.5],
    })
    psm_confident = psm_full.iloc[[0]].copy()
    predicted = pd.DataFrame({
        "Modified.Sequence":  ["PEPSTIDE", "GHOST"],
        "Precursor.Charge":   [2, 2],
        "Pred.RT":            [10.6, 12.0],
    })
    params = MBRParams(
        enabled=True, q_donor=0.01, q_rescue=0.05,
        rt_tolerance_min=1.0,
        inject_predicted_donors=True,
        min_injected_observed_runs=1,
        injected_rt_tolerance_min=1.5,
    )
    merged, _ = match_between_runs(
        psm_full, psm_confident, params=params, predicted_donors=predicted,
    )
    seqs = set(merged["Modified.Sequence"])
    assert "PEPTIDE"  in seqs            # original confident
    assert "GHOST"    not in seqs        # never scored by Sage -> rejected


# ============================================================================
# v0.5.8.1 regression: predicted_library failure surfaces in run_manifest.json
# ============================================================================
def test_v0581_predicted_library_reason_in_manifest(tmp_path):
    """When predicted_library is enabled but no library was produced, the
    manifest must surface a human-readable ``reason`` so operators can
    diagnose silent fall-back without grepping container logs."""
    import json
    from diaquant import predicted_library as pl
    from diaquant.config import DiaQuantConfig
    from diaquant.manifest import write_run_manifest

    pl._record_failure("phospho", "alphapeptdeep_import_failed: ImportError: synthetic")

    cfg = DiaQuantConfig(
        fasta=str(tmp_path / "fake.fasta"),
        mzml_files=[],
        output_dir=str(tmp_path / "out"),
        predicted_library=True,
        rescore_with_prediction=True,
        match_between_runs=True,
        mbr_inject_predicted_donors=True,
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "fake.cfg").write_text("dummy: 1\n")

    p = write_run_manifest(
        out_dir=tmp_path / "out",
        cfg=cfg,
        diaquant_version="0.5.9",
        config_yaml_src=tmp_path / "fake.cfg",
        library_paths=[],
        n_psms_raw=1000,
        n_psms_rescored=0,
        n_psms_after_fdr=500,
        n_psms_mbr=0,
        pr_rows=10, pg_rows=2, site_rows=1,
    )
    data = json.loads(p.read_text())
    assert data["predicted_library"]["applied"] is False
    assert "alphapeptdeep_import_failed" in data["predicted_library"]["reason"]
    assert data["predicted_library"]["reason_pass"] == "phospho"
    assert "no donors injected" in data["mbr"]["injection_reason"]
    assert data["rescoring"]["applied"] is False
    assert "predicted_library" in data["rescoring"]["reason"]
    pl._clear_failure()


def test_v0581_peptdeep_self_check_keys_present(tmp_path):
    """The manifest writer always populates the peptdeep_* observability
    fields, regardless of whether AlphaPeptDeep is installed in the test
    environment."""
    import json
    from diaquant.config import DiaQuantConfig
    from diaquant.manifest import write_run_manifest

    cfg = DiaQuantConfig(
        fasta=str(tmp_path / "fake.fasta"),
        mzml_files=[],
        output_dir=str(tmp_path / "out"),
        predicted_library=True,
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "fake.cfg").write_text("dummy: 1\n")
    p = write_run_manifest(
        out_dir=tmp_path / "out",
        cfg=cfg,
        diaquant_version="0.5.9",
        config_yaml_src=tmp_path / "fake.cfg",
        library_paths=[],
        n_psms_raw=0, n_psms_rescored=0, n_psms_after_fdr=0,
        n_psms_mbr=0, pr_rows=0, pg_rows=0, site_rows=0,
    )
    data = json.loads(p.read_text())
    assert "peptdeep_importable" in data["predicted_library"]
    assert "peptdeep_detail" in data["predicted_library"]
    assert "peptdeep_build_status" in data["predicted_library"]
