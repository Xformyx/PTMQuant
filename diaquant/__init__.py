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

v0.5.8 changes (the "identification sensitivity" release):
  P0.   Predicted-library MBR donor injection.  AlphaPeptDeep predicted
        precursors are now added to the MBR donor pool with FDR-safe
        gating: each injected donor must have been scored by Sage at any
        q-value in at least ``mbr_min_injected_observed_runs`` (default 1)
        runs, so target-decoy FDR is preserved (no PSM is invented).
        Closes the recall gap that left v0.5.7 at 50.6% UniProt-accession
        concordance vs. DIA-NN on KIST-EPS phospho-DIA.
  P1-a. Per-pass peptide_fdr.  All PTM-aware built-in profiles (phospho,
        ubiquitin, acetyl_methyl, succinyl_acyl, oglcnac, citrullination,
        lactyl_acyl) override peptide_fdr to 0.05 because the FDR
        estimator is dominated by unmodified peptides; truncating the
        rare modified hits at 1% q-value was cutting real phospho rows.
        site_probability_cutoff stays at 0.75 as the localisation
        safeguard, and whole_proteome still runs at 1% global.
  P1-b. Group-aware imputation (``imputation.py``).  New
        ``ImputeParams`` + ``impute_matrix`` apply per-condition median /
        min / KNN imputation to pr_matrix / pg_matrix /
        ptm_site_matrix, but never invent values for groups with fewer
        than ``impute_min_obs_per_group`` valid observations.  A new
        ``Intensity.Imputed.Frac`` column lets downstream tools
        re-filter rows whose imputation proportion is too high.  Default
        ``impute_method: "none"`` preserves v0.5.7 behaviour.

v0.5.7 changes (the "v0.5.6 hotfix + precursor-normalization" release):
  P0-1. parse_sage_tsv now drops Sage decoys (label == -1, plus rev_/REV_/
        DECOY_ accession-prefixed rows) **before** the FDR filter.  v0.5.6
        leaked 185 decoy precursors and 101 decoy protein groups into the
        DIA-NN-style outputs because the q-value filter alone does not
        guarantee target-only rows.
  P0-2. site_quant() no longer iterates with ``DataFrame.itertuples``,
        which silently mangles columns with dotted names (e.g. ``PTM.Mods``
        becomes ``_2``) and caused every PSM to be skipped, leaving
        ``ptm_site_matrix.tsv`` empty.  The PTM.Mods path now uses
        per-column numpy arrays and emits a real, populated site matrix.
  P2-1. New ``precursor_matrix_normalized()`` runs directLFQ's
        NormalizationManager on the precursor pivot before writing
        ``report.pr_matrix.tsv``, equalising per-sample medians in log
        space.  This drops the missing-value rate from the v0.5.6 38.9%
        back to the expected ~18% range when KIST-EPS-style runs differ
        in load by 1.5--3x.  Toggle via ``normalize_precursor_matrix``
        in YAML (default true).

v0.5.6 changes (the "release engineering" release):
  * Dockerfile hardened: non-root ``ptmq`` user (uid 1000), OCI labels
    carrying ``org.opencontainers.image.version == __version__``, HEALTH-
    CHECK exercising ``diaquant --version``, shared predicted-library
    cache mounted at ``/cache/predicted_libs`` matching the
    ``PTMQUANT_LIB_CACHE_DIR`` default, ``PEPTDEEP_STRICT`` build arg,
    and ``/etc/ptmquant/peptdeep_status.txt`` recording whether the
    pretrained models were baked in.
  * ``pyproject.toml`` now reads ``version`` dynamically from
    ``diaquant.__version__`` so the image / wheel / ``diaquant
    --version`` / ``run_manifest.json`` cannot disagree.
  * New GHCR publishing workflow (``.github/workflows/docker-publish``)
    builds and pushes ``ghcr.io/xformyx/ptmquant:<version>`` plus
    ``:latest`` on every ``v*.*.*`` tag, so re-running the pipeline
    after a release is a single ``docker pull``.
  * New ``scripts/verify_ptmquant.py``: stdlib-only post-hoc verifier
    that checks ``run_manifest.json`` exists, the diaquant version
    advertised inside it is recent enough, Genes are populated in
    ``pg_matrix``, ``pg_matrix`` covers a sensible fraction of
    ``pr_matrix`` accessions, and (advisory) MBR / predicted-library
    / ptm_site_matrix are healthy.  Returns exit codes suitable for
    CI gating.

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

__version__ = "0.5.8"
