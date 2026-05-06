"""Phase 3 (v0.6.0) — tests for the AlphaDIA precursor parser and the
DIA-NN-style ``Modified.Sequence`` formatter.

Two thin layers are validated:

1. :func:`diaquant.modifications.format_diann_sequence` — the pure formatter
   that turns AlphaDIA's ``(sequence, mods, mod_sites)`` triple into the
   DIA-NN ``_AAS(UniMod:21)PEPTIDER_`` string the rest of PTMQuant expects.
   Coverage spans unmodified peptides, single PTMs, multi-PTM peptides,
   N-/C-terminal placements, custom (non-UniMod) modifications and the
   degenerate ``mods != mod_sites`` length-mismatch case.

2. :func:`diaquant.parse_alphadia.parse_alphadia_precursor` — the end-to-end
   parser run against an in-memory AlphaDIA precursor frame.  We assert that
   the standard PTMQuant downstream contract is honoured: ``filename``,
   ``Modified.Sequence`` (UniMod tokens), ``Stripped.Sequence``, ``Precursor.Charge``,
   ``Precursor.Id``, ``Protein.Group``, ``Genes``, ``RT`` (minutes),
   ``Peptide.Q.Value``, ``Intensity`` and the localisation columns added by
   :mod:`diaquant.ptm_localization`.  Decoy filtering and the optional
   secondary FDR filter are also covered.
"""

from __future__ import annotations

import pandas as pd
import pytest

from diaquant.modifications import format_diann_sequence
from diaquant.parse_alphadia import (
    ALPHADIA_TO_DIANN,
    _build_modified_sequence,
    _strip_mods,
    parse_alphadia_precursor,
)


# ---------------------------------------------------------------------------
# format_diann_sequence
# ---------------------------------------------------------------------------

class TestFormatDiannSequence:

    def test_unmodified_peptide_just_gets_underscores(self):
        assert format_diann_sequence("PEPTIDER", "", "") == "_PEPTIDER_"

    def test_single_phospho_on_serine(self):
        # Phospho == UniMod:21
        out = format_diann_sequence("AASPEPTIDER", "Phospho@S", "3")
        assert out == "_AAS(UniMod:21)PEPTIDER_"

    def test_oxidation_on_methionine(self):
        out = format_diann_sequence("PEPMTIDE", "Oxidation@M", "4")
        assert out == "_PEPM(UniMod:35)TIDE_"

    def test_glygly_on_lysine(self):
        # GlyGly (di-glycine ubiquitin remnant) == UniMod:121
        out = format_diann_sequence("PEPTIDEK", "GlyGly@K", "8")
        assert out == "_PEPTIDEK(UniMod:121)_"

    def test_protein_n_term_acetyl(self):
        # mod_site == 0 -> N-term, before the first residue
        out = format_diann_sequence("MPEPTIDE", "Acetyl@Protein_N-term", "0")
        assert out == "_(UniMod:1)MPEPTIDE_"

    def test_multi_ptm_peptide_with_nterm(self):
        # All three mods on one peptide -- N-term Acetyl + Met oxidation +
        # Phospho on Ser at position 9.
        out = format_diann_sequence(
            "MPEPTIDES",
            "Acetyl@Protein_N-term;Oxidation@M;Phospho@S",
            "0;1;9",
        )
        assert out == "_(UniMod:1)M(UniMod:35)PEPTIDES(UniMod:21)_"

    def test_unknown_mod_falls_back_to_name_token(self):
        # Custom name with no UniMod accession: keep the readable name in
        # parentheses so downstream parsers don't drop the modification.
        out = format_diann_sequence("PEPTIDEK", "MyCustomMod@K", "8")
        assert out == "_PEPTIDEK(MyCustomMod)_"

    def test_length_mismatch_returns_bare_sequence(self):
        # mods has 2 entries, mod_sites has 1 -> defensive fallback
        out = format_diann_sequence("PEPTIDEK", "Phospho;GlyGly", "8")
        assert out == "_PEPTIDEK_"

    def test_bare_mod_name_without_at_target(self):
        # AlphaDIA may emit just "Phospho" instead of "Phospho@S" for some
        # library variants; the formatter must still resolve UniMod:21.
        out = format_diann_sequence("AASPEPTIDER", "Phospho", "3")
        assert out == "_AAS(UniMod:21)PEPTIDER_"

    def test_methyl_on_arginine(self):
        out = format_diann_sequence("PEPTIDER", "Methyl@R", "8")
        assert out == "_PEPTIDER(UniMod:34)_"


# ---------------------------------------------------------------------------
# parse_alphadia_precursor
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_alphadia_precursor(tmp_path):
    """Return the path to a synthetic AlphaDIA precursor.parquet table.

    The frame intentionally mixes target / decoy hits, modified / unmodified
    peptides and runs across two raw files so we can assert all the
    downstream-relevant code paths in one place.
    """
    df = pd.DataFrame({
        "run":                    ["s1", "s1", "s2", "s2", "s1"],
        "precursor.sequence":     ["AASPEPTIDER", "MPEPTIDE", "PEPTIDEK",
                                    "VANILLAPEP", "PEPTIDEK"],
        "precursor.mods":         ["Phospho@S",
                                    "Acetyl@Protein_N-term;Oxidation@M",
                                    "GlyGly@K", "", "GlyGly@K"],
        "precursor.mod_sites":    ["3", "0;1", "8", "", "8"],
        "precursor.charge":       [2, 3, 2, 2, 2],
        "precursor.qval":         [0.001, 0.01, 0.04, 0.0005, 0.20],  # last one above 5%
        "precursor.proba":        [0.99, 0.95, 0.80, 0.99, 0.40],
        "precursor.intensity":    [1.0e7, 5.0e6, 2.0e6, 8.0e6, 1.0e5],
        "precursor.rt.observed":  [1800.0, 2400.0, 1500.0, 600.0, 1500.0],
        "precursor.rt.library":   [1810.0, 2410.0, 1490.0, 605.0, 1490.0],
        "pg.name":                ["sp|P0001|GENE1", "sp|P0002|GENE2",
                                    "sp|P0003|GENE3", "sp|P0004|GENE4",
                                    "rev_sp|P0099|DECOY"],
        "pg.proteins":            ["sp|P0001|GENE1", "sp|P0002|GENE2",
                                    "sp|P0003|GENE3", "sp|P0004|GENE4",
                                    "rev_sp|P0099|DECOY"],
        "pg.genes":               ["GENE1", "GENE2", "GENE3", "GENE4", "DECOY"],
        "pg.qval":                [0.005, 0.01, 0.02, 0.001, 0.50],
    })
    p = tmp_path / "precursor.parquet"
    df.to_parquet(p, index=False)
    return p


class TestParseAlphaDiaPrecursor:

    def test_returns_dataframe_with_standard_columns(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        required = {
            "filename", "Modified.Sequence", "Stripped.Sequence",
            "Precursor.Charge", "Precursor.Id",
            "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
            "First.Protein.Description", "Proteotypic",
            "RT", "Predicted.RT",
            "Peptide.Q.Value", "Protein.Q.Value",
            "Intensity", "Score",
        }
        missing = required - set(df.columns)
        assert not missing, f"missing PTMQuant standard columns: {missing}"

    def test_modified_sequence_uses_unimod_tokens(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        # Find the phospho row (s1, AASPEPTIDER, charge 2)
        row = df[(df["filename"] == "s1") & (df["Stripped.Sequence"] == "AASPEPTIDER")]
        assert len(row) == 1
        assert row.iloc[0]["Modified.Sequence"] == "_AAS(UniMod:21)PEPTIDER_"

    def test_protein_n_term_acetyl_round_trips(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        row = df[df["Stripped.Sequence"] == "MPEPTIDE"]
        assert len(row) == 1
        # Acetyl-Nterm (UniMod:1) + Met oxidation (UniMod:35)
        assert row.iloc[0]["Modified.Sequence"] == "_(UniMod:1)M(UniMod:35)PEPTIDE_"

    def test_unmodified_peptide_is_underscore_wrapped(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        row = df[df["Stripped.Sequence"] == "VANILLAPEP"]
        assert len(row) == 1
        assert row.iloc[0]["Modified.Sequence"] == "_VANILLAPEP_"

    def test_decoy_rows_are_dropped(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        # The synthetic frame had one rev_ row; it must not survive.
        assert not df["Protein.Group"].astype(str).str.startswith("rev_").any()
        # 5 input rows -> 4 after decoy drop
        assert len(df) == 4

    def test_rt_is_converted_to_minutes(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        # The phospho row had RT=1800s -> must become 30.0 min.
        row = df[df["Stripped.Sequence"] == "AASPEPTIDER"]
        assert row.iloc[0]["RT"] == pytest.approx(30.0)
        assert row.iloc[0]["Predicted.RT"] == pytest.approx(1810.0 / 60.0)

    def test_precursor_id_is_seq_plus_charge(self, fake_alphadia_precursor):
        df = parse_alphadia_precursor(fake_alphadia_precursor)
        row = df[df["Stripped.Sequence"] == "AASPEPTIDER"].iloc[0]
        assert row["Precursor.Id"] == "_AAS(UniMod:21)PEPTIDER_2"

    def test_optional_fdr_filter_applies(self, fake_alphadia_precursor):
        # peptide_fdr=0.05 should drop any row with q>0.05 (the decoy is
        # already gone, so no rows should be lost in this fixture).
        df = parse_alphadia_precursor(fake_alphadia_precursor, peptide_fdr=0.05)
        assert (df["Peptide.Q.Value"].astype(float) <= 0.05).all()

    def test_return_unfiltered_returns_two_frames(self, fake_alphadia_precursor):
        df_filt, df_full = parse_alphadia_precursor(
            fake_alphadia_precursor, peptide_fdr=0.01, return_unfiltered=True
        )
        assert isinstance(df_filt, pd.DataFrame)
        assert isinstance(df_full, pd.DataFrame)
        # df_full carries the donor pool -- it must be at least as large as
        # the filtered table.
        assert len(df_full) >= len(df_filt)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_strip_mods_removes_unimod_tags(self):
        assert _strip_mods("_AAS(UniMod:21)PEPTIDER_") == "AASPEPTIDER"
        assert _strip_mods("_(UniMod:1)M(UniMod:35)PEPTIDES(UniMod:21)_") \
            == "MPEPTIDES"

    def test_strip_mods_handles_mass_delta_syntax(self):
        # Sage-style mass delta also gets stripped (parse_sage compatibility).
        assert _strip_mods("AAS[+79.9663]PEPTIDER") == "AASPEPTIDER"

    def test_alphadia_to_diann_renames_cover_required_columns(self):
        # Just a sanity check that the rename map didn't drop a key column.
        targets = set(ALPHADIA_TO_DIANN.values())
        for required in (
            "filename", "Stripped.Sequence", "Precursor.Charge",
            "Peptide.Q.Value", "Intensity", "RT",
            "Protein.Group", "Protein.Ids", "Genes",
        ):
            assert required in targets, f"{required} missing from rename map"

    def test_build_modified_sequence_handles_nan_strings(self):
        df = pd.DataFrame({
            "precursor.sequence": ["AAA", "BBB"],
            "precursor.mods":     ["", "Phospho@S"],
            "precursor.mod_sites": ["", "1"],
        })
        out = _build_modified_sequence(df)
        assert out.iloc[0] == "_AAA_"
        assert out.iloc[1] == "_B(UniMod:21)BB_"
