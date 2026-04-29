"""diaquant: open-source DIA proteomics quantification pipeline.

Modules:
- config:             parse YAML configuration
- enzymes:            9-entry protease catalog (NEW in 0.5.1)
- instruments:        4-entry Orbitrap preset catalog (NEW in 0.5.1)
- modifications:      UniMod / custom PTM definitions
- ptm_profiles:       PTM-family-specific Sage search profiles (multi-pass)
- sage_runner:        build Sage JSON config and invoke the binary
- multipass:          run several PTM passes and merge their results; v0.5.5
                      can return the unfiltered PSM pool needed for MBR
- predicted_library:  AlphaPeptDeep predicted spectral library (0.5.0);
                      cross-job shared cache via ``pred_lib_cache_dir`` /
                      ``PTMQUANT_LIB_CACHE_DIR`` (NEW in 0.5.3)
- rescore:            post-hoc AlphaPeptDeep-based PSM rescoring (0.5.0)
- ptm_localization:   site-level PTM probability scoring; v0.5.5 splits
                      positions per-mod (``PTM.Mods``), normalises mass tags
                      (+79.9663 -> Phospho), and fixes the softmax=1.0 pitfall
- mbr:                v0.5.5 match-between-runs rescue based on RT-aligned
                      cross-run donor anchors + Pred.RT agreement
- quantify:           directLFQ wrapper (precursor -> protein, precursor -> site);
                      v0.5.5 site_quant consumes PTM.Mods and filters per-mod
- rt_align:           LOWESS run-to-run retention time alignment; v0.5.5
                      adds a dedicated phospho-only LOWESS overlay and
                      Pred.RT-based anchor filtering (tol=2 min)
- stats:              sample-sheet-driven differential analysis
- writer:             DIA-NN compatible TSV emitters
- razor:              v0.5.4 Occam razor protein grouping
- manifest:           v0.5.4 run_manifest writer; v0.5.5 auto-discovers
                      predicted_library_*.tsv in pass subdirs + emits a
                      stub manifest on exception
- cli:                click-based command-line interface

v0.5.5 changes (the "observability + PTM" release):
  P0-A. ptm_site_matrix 0-rows bug fixed.  ``add_site_probabilities`` now
        records per-mod positions (``PTM.Mods`` dict) and normalises mass
        tags into canonical names so the phospho whitelist matches.
  P0-B. Match-between-runs (``mbr.py``): borderline PSMs in run B that
        match a confident donor's peptide+charge within RT tolerance are
        rescued, cutting precursor missing-value rate while preserving FDR.
  P1-A. Phospho-aware RT alignment: a second LOWESS curve is fit on
        phospho PSMs and applied only to phospho rows.  Pred.RT anchor
        filter (|RT - Pred.RT| <= 2 min) removes outlier anchors that
        were dragging the curve.
  P1-B. run_manifest.json is now emitted even when write fails (stub with
        traceback), and auto-discovers predicted_library_*.tsv in
        ``pass_phospho/`` / cache dir so multi-pass outputs are visible.
"""

__version__ = "0.5.5"
