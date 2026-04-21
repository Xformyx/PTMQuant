"""diaquant: open-source DIA proteomics quantification pipeline.

Modules:
- config:           parse YAML configuration
- modifications:    UniMod / custom PTM definitions
- ptm_profiles:     PTM-family-specific Sage search profiles (multi-pass)
- sage_runner:      build Sage JSON config and invoke the binary
- multipass:        run several PTM passes and merge their results
- ptm_localization: site-level PTM probability scoring
- quantify:         directLFQ wrapper (precursor -> protein, precursor -> site)
- rt_align:         LOWESS run-to-run retention time alignment (always-on)
- stats:            sample-sheet-driven differential analysis
- writer:           DIA-NN compatible TSV emitters
- cli:              click-based command-line interface
"""

__version__ = "0.4.0"
