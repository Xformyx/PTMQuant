"""v0.6.0a5 Phase 4 tests — `diaquant run --engine={alphadia|sage}`.

These tests verify the new `--engine` flag, the default-to-alphadia
behaviour, and the `_run_search_alphadia` dispatcher's ability to:

  * iterate over multi-pass profiles,
  * write per-pass output directories,
  * call ``run_alphadia`` with the correct ``PassProfile`` per pass,
  * concatenate per-pass DataFrames with a ``Pass`` column,
  * honour ``--resume`` by skipping subprocess invocation when a
    cached precursor file exists,
  * surface AlphaDIA failures as Click errors (no silent Sage fallback).

We monkey-patch ``run_alphadia`` and ``parse_alphadia_precursor`` so
these tests run in milliseconds and require neither the AlphaDIA
binary nor real mzML inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from diaquant.cli import cli, _run_search_alphadia
from diaquant.alphadia_runner import AlphaDIAError
from diaquant.config import DiaQuantConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_cfg(tmp_path: Path) -> DiaQuantConfig:
    """Build a DiaQuantConfig that points to throw-away tmp paths.

    We never actually run a search; the dispatcher only inspects ``cfg`` to
    pick passes and emit log lines.
    """
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">sp|P00000|TEST_HUMAN\nMAGICPEPTIDER\n")
    mzml = tmp_path / "sample_01.mzML"
    mzml.write_text("<mzML/>")
    out = tmp_path / "out"
    out.mkdir()
    cfg = DiaQuantConfig(
        fasta=fasta,
        mzml_files=[mzml],
        output_dir=out,
        passes=[],
        custom_passes=[],
        peptide_fdr=0.01,
        site_probability_cutoff=0.75,
    )
    return cfg


def _fake_precursor_df() -> pd.DataFrame:
    """Tiny PTMQuant-schema PSM frame, just enough for the dispatcher to
    concatenate and tag with ``Pass``.
    """
    return pd.DataFrame({
        "filename": ["sample_01"],
        "Stripped.Sequence": ["MAGICPEPTIDER"],
        "Modified.Sequence": ["_MAGICPEPTIDER_"],
        "Precursor.Charge": [2],
        "Precursor.Id": ["MAGICPEPTIDER2"],
        "Protein.Group": ["P00000"],
        "Protein.Ids": ["P00000"],
        "Genes": ["TEST"],
        "Q.Value": [0.001],
        "Peptide.Q.Value": [0.001],
        "Protein.Q.Value": [0.001],
        "MS1.Intensity": [1.0e6],
        "Intensity": [1.0e6],
        "Score": [0.99],
        "RT (min)": [12.3],
        "Predicted.RT (min)": [12.5],
        "Calc.Mass": [1500.7],
        "Exp.Mass": [1500.71],
        "Proteotypic": [1],
    })


# ---------------------------------------------------------------------------
# Tests: --engine flag presence and defaults
# ---------------------------------------------------------------------------

def test_run_help_lists_engine_flag():
    runner = CliRunner()
    res = runner.invoke(cli, ["run", "--help"])
    assert res.exit_code == 0, res.output
    assert "--engine" in res.output
    assert "alphadia" in res.output
    # default must be alphadia per Phase 4 contract
    assert "[default: alphadia]" in res.output


def test_run_help_lists_library_flag():
    runner = CliRunner()
    res = runner.invoke(cli, ["run", "--help"])
    assert res.exit_code == 0, res.output
    assert "--library" in res.output


# ---------------------------------------------------------------------------
# Tests: _run_search_alphadia dispatcher behaviour (monkey-patched)
# ---------------------------------------------------------------------------

def test_dispatcher_single_pass_calls_run_alphadia_once(
    monkeypatch, minimal_cfg, tmp_path
):
    """When cfg has no passes, dispatcher must call run_alphadia exactly
    once with pass_profile=None (single-pass mode).
    """
    calls = []

    def fake_run_alphadia(cfg, pass_profile=None, library_path=None,
                          output_dir=None, **kw):
        calls.append({"pass": pass_profile, "out": output_dir})
        # Simulate AlphaDIA writing a precursor.tsv
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "precursor.tsv").write_text("dummy")
        return Path(output_dir)

    def fake_parse(precursor_path, **kw):
        return _fake_precursor_df()

    monkeypatch.setattr("diaquant.cli.__getattr__", lambda n: None,
                        raising=False)
    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", fake_run_alphadia
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.parse_alphadia_precursor", fake_parse
    )
    # attach_fasta_meta should be a passthrough for test isolation
    monkeypatch.setattr(
        "diaquant.parse_alphadia.attach_fasta_meta",
        lambda df, fasta: df,
    )

    df = _run_search_alphadia(minimal_cfg, resume=False, library_override=None)

    assert len(calls) == 1, f"expected 1 call, got {len(calls)}"
    assert calls[0]["pass"] is None, "single-pass must pass profile=None"
    assert "alphadia/single" in str(calls[0]["out"]).replace("\\", "/")
    # Result frame must have Pass column tagged 'single'
    assert "Pass" in df.columns
    assert (df["Pass"] == "single").all()
    # engine attr must be set for run_manifest
    assert df.attrs.get("engine") == "alphadia"


def test_dispatcher_multi_pass_iterates_all_passes(
    monkeypatch, minimal_cfg, tmp_path
):
    """With cfg.passes=[phospho, ubiquitin], the dispatcher must invoke
    run_alphadia twice (one per pass) and concatenate the two frames
    with the appropriate Pass column.
    """
    minimal_cfg.passes = ["phospho", "ubiquitin"]
    calls = []

    def fake_run_alphadia(cfg, pass_profile=None, library_path=None,
                          output_dir=None, **kw):
        calls.append(getattr(pass_profile, "name", "?"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "precursor.tsv").write_text("dummy")
        return Path(output_dir)

    def fake_parse(precursor_path, **kw):
        return _fake_precursor_df()

    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", fake_run_alphadia
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.parse_alphadia_precursor", fake_parse
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.attach_fasta_meta",
        lambda df, fasta: df,
    )

    df = _run_search_alphadia(minimal_cfg, resume=False, library_override=None)

    assert calls == ["phospho", "ubiquitin"]
    assert set(df["Pass"].unique()) == {"phospho", "ubiquitin"}
    assert len(df) == 2  # one row per pass


def test_dispatcher_resume_skips_subprocess_when_cache_exists(
    monkeypatch, minimal_cfg, tmp_path
):
    """When --resume is on AND a precursor file already exists in the
    pass output dir, run_alphadia must NOT be called.
    """
    minimal_cfg.passes = ["phospho"]
    cached = minimal_cfg.output_dir / "alphadia" / "phospho"
    cached.mkdir(parents=True, exist_ok=True)
    (cached / "precursor.tsv").write_text("cached")

    called = {"n": 0}

    def fake_run_alphadia(*a, **kw):
        called["n"] += 1
        return Path(kw.get("output_dir") or "")

    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", fake_run_alphadia
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.parse_alphadia_precursor",
        lambda p, **kw: _fake_precursor_df(),
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.attach_fasta_meta",
        lambda df, fasta: df,
    )

    df = _run_search_alphadia(minimal_cfg, resume=True, library_override=None)
    assert called["n"] == 0, "resume must skip subprocess when cache exists"
    assert len(df) == 1


def test_dispatcher_propagates_alphadia_failure(
    monkeypatch, minimal_cfg
):
    """An AlphaDIAError from run_alphadia must surface as a ClickException
    (NOT silently swallowed nor falling back to Sage).
    """
    minimal_cfg.passes = ["phospho"]

    def boom(*a, **kw):
        raise AlphaDIAError("simulated subprocess crash")

    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", boom
    )

    with pytest.raises(click.ClickException) as exc_info:
        _run_search_alphadia(minimal_cfg, resume=False, library_override=None)
    assert "phospho" in str(exc_info.value.message)
    assert "simulated subprocess crash" in str(exc_info.value.message)


def test_dispatcher_unknown_pass_name_errors_clearly(
    monkeypatch, minimal_cfg
):
    minimal_cfg.passes = ["definitely_not_a_real_pass"]
    with pytest.raises(click.ClickException) as exc_info:
        _run_search_alphadia(minimal_cfg, resume=False, library_override=None)
    assert "Unknown built-in pass" in str(exc_info.value.message)


def test_dispatcher_records_library_override_in_attrs(
    monkeypatch, minimal_cfg, tmp_path
):
    lib = tmp_path / "predicted.tsv"
    lib.write_text("library\tcontent")

    def fake_run_alphadia(cfg, pass_profile=None, library_path=None,
                          output_dir=None, **kw):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "precursor.tsv").write_text("dummy")
        # Library path must be propagated end-to-end
        assert library_path is not None
        return Path(output_dir)

    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", fake_run_alphadia
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.parse_alphadia_precursor",
        lambda p, **kw: _fake_precursor_df(),
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.attach_fasta_meta",
        lambda df, fasta: df,
    )

    df = _run_search_alphadia(
        minimal_cfg, resume=False, library_override=str(lib)
    )
    assert df.attrs["predicted_library_paths"] == [str(lib.resolve())]
    assert df.attrs["engine"] == "alphadia"


def test_dispatcher_per_pass_fdr_is_threaded_through(
    monkeypatch, minimal_cfg, tmp_path
):
    """The phospho pass profile sets peptide_fdr=0.05; that exact value
    must reach parse_alphadia_precursor.
    """
    minimal_cfg.passes = ["phospho"]
    captured = {}

    def fake_run_alphadia(cfg, pass_profile=None, library_path=None,
                          output_dir=None, **kw):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "precursor.tsv").write_text("dummy")
        return Path(output_dir)

    def fake_parse(precursor_path, *, site_cutoff, peptide_fdr, **kw):
        captured["site_cutoff"] = site_cutoff
        captured["peptide_fdr"] = peptide_fdr
        return _fake_precursor_df()

    monkeypatch.setattr(
        "diaquant.alphadia_runner.run_alphadia", fake_run_alphadia
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.parse_alphadia_precursor", fake_parse
    )
    monkeypatch.setattr(
        "diaquant.parse_alphadia.attach_fasta_meta",
        lambda df, fasta: df,
    )

    _run_search_alphadia(minimal_cfg, resume=False, library_override=None)
    # Phospho pass profile: peptide_fdr=0.05, site_probability_cutoff=0.75
    assert captured["peptide_fdr"] == 0.05
    assert captured["site_cutoff"] == 0.75


def test_default_engine_choice_is_alphadia_in_signature():
    """Direct introspection of the click Command default — guards against
    accidental flips of the default back to sage.
    """
    from diaquant.cli import run as run_cmd
    engine_param = next(
        p for p in run_cmd.params if getattr(p, "name", None) == "engine"
    )
    assert engine_param.default == "alphadia"
    assert "alphadia" in engine_param.type.choices
    assert "sage" in engine_param.type.choices
