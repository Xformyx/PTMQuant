"""diaquant: open-source DIA proteomics quantification pipeline.

Modules:
- config:             parse YAML configuration
- enzymes:            9-entry protease catalog (NEW in 0.5.1)
- instruments:        4-entry Orbitrap preset catalog (NEW in 0.5.1)
- modifications:      UniMod / custom PTM definitions
- ptm_profiles:       PTM-family-specific Sage search profiles (multi-pass)
- sage_runner:        build Sage JSON config and invoke the binary
- multipass:          run several PTM passes and merge their results
- predicted_library:  AlphaPeptDeep predicted spectral library (0.5.0);
                      cross-job shared cache via ``pred_lib_cache_dir`` /
                      ``PTMQUANT_LIB_CACHE_DIR`` (NEW in 0.5.3)
- rescore:            post-hoc AlphaPeptDeep-based PSM rescoring (0.5.0)
- ptm_localization:   site-level PTM probability scoring
- quantify:           directLFQ wrapper (precursor -> protein, precursor -> site)
                      v0.5.3 site_quant computes absolute protein positions and
                      honours ``localization_cutoff`` for the phospho filter.
- rt_align:           LOWESS run-to-run retention time alignment (always-on)
- stats:              sample-sheet-driven differential analysis
- writer:             DIA-NN compatible TSV emitters; 0.5.3 site matrix now
                      emits Protein.Group, Genes, PTM.Site (absolute position)
                      and Best.Site.Probability columns.
- cli:                click-based command-line interface
"""

__version__ = "0.5.3.1"
