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

v0.5.9.2 changes (the "directLFQ-dask quantification hotfix" release):
  P0.   directLFQ-based protein and site quantification now runs with
        ``dask`` parallelism on a worker pool sized by
        ``cfg.quant_num_cores`` / ``PTMQUANT_QUANT_CORES`` /
        ``os.cpu_count()`` (capped at 16).  Without this, v0.5.9.1
        completed every step except the final 5%, where
        ``estimate_protein_intensities`` ran single-threaded against the
        much larger PSM pool that v0.5.9.1's working AlphaPeptDeep
        predicted_library produced.  A 12-mzML phospho job on KIST-EPS
        consumed >15 hours and 56 GB RAM at 50% CPU before the operator
        stopped it.  ``num_cores`` is now resolved up-front and passed
        explicitly to ``estimate_protein_intensities`` instead of relying
        on directLFQ's silent fallback to single-threaded when dask is
        absent.  Expected speedup on a 16-core Mac Studio: 5-10x with
        peak RSS dropping from 56 GB to ~10 GB because dask processes
        proteins in chunks.
  P0.   The Docker image now installs ``dask[dataframe]>=2024.5`` and
        runs a ``import dask, dask.dataframe`` smoke test so a
        regression of this acceleration cannot reach GHCR.  When the
        host pip image is built without dask, the runtime emits a loud
        warning at every directLFQ call.
  P1.   Added ``[quantify-protein|site|pr_norm]`` progress logs at
        start / after normalisation / after estimation with elapsed
        wall-clock seconds and the resolved core count.  Operators can
        now distinguish "hung" from "slow" without attaching py-spy.
  P1.   ``cfg.quant_num_cores`` (env override
        ``PTMQUANT_QUANT_CORES``) lets the operator clamp the worker
        pool when the host is shared with other workloads.

v0.5.9.1 changes (the "chunked-predict OOM hotfix" release):
  P0.   AlphaPeptDeep ``predict_all`` is now executed in chunks of
        ``cfg.pred_lib_chunk_size`` precursors (default 1_000_000) instead
        of one giant pass.  v0.5.9 produced a working library on small
        FASTAs but caught SIGKILL (exit 137) on whole-proteome multi-PTM
        jobs because peptdeep allocates two ``(n_prec, max_frag_idx, 8)``
        float32 matrices for ``fragment_mz_df`` / ``fragment_intensity_df``
        in a single shot -- ~40 GB at 23 M precursors plus pandas
        overhead, exceeding even a 96 GB Docker Desktop allocation on a
        Mac Studio.  v0.5.9.1 calls ``ModelManager.predict_all`` directly
        on precursor_df slices, writes each chunk to its own TSV via
        ``translate_to_tsv``, then drops every reference and runs
        ``gc.collect()`` before moving on.  Peak resident set drops from
        ~80 GB to ~5-6 GB regardless of FASTA / PTM combinatorics.
  P0.   The chunk TSVs are stream-concatenated into the canonical
        ``predicted_library_<hash>.tsv`` (chunk 1 verbatim, chunks 2..N
        with their header line stripped) so downstream Sage / multipass
        consumers see a byte-identical file to the legacy single-shot
        path; cache hash + manifest plumbing are unchanged.
  P0.   Hard-fail policy: when the host has less than
        ``pred_lib_memory_budget_gb`` available or the digested precursor
        count exceeds ``pred_lib_max_precursors``, the pipeline now
        raises ``MemoryError`` instead of silently falling back to the
        in-silico library.  This was an explicit user request after
        v0.5.9: better to surface the failure than to ship a degraded
        library.  ``pred_lib_fallback_in_silico`` default flips from
        ``True`` to ``False`` accordingly; set it back to ``True`` for
        debugging only.
  P1.   ``pred_lib_max_precursors`` raised from 2 M -> 50 M, since the
        chunked path is what bounds memory now.  The cap exists only to
        refuse pathological inputs (e.g. metaproteome + max_var=5 ->
        100 M+ precursors).
  P1.   New env-var override ``PTMQUANT_PEPTDEEP_CHUNK`` for the chunk
        size (alongside the existing ``PTMQUANT_PEPTDEEP_BATCH`` and
        ``PTMQUANT_PEPTDEEP_MAX_PRECURSORS``).
  P1.   Per-chunk progress logging shows ``rows lo..hi`` plus available
        memory after each chunk via psutil, so OOM behaviour is now
        directly observable in the platform UI without attaching a
        debugger.

v0.5.9 changes (the "AlphaPeptDeep ABI hotfix" release):
  P0.   Pin ``transformers==4.47.0`` (plus ``numba==0.60.0`` and
        ``numpy<2``) in both the Docker image and the ``[deeplearning]``
        extra.  AlphaPeptDeep was failing to import at runtime with
        ``ImportError: cannot import name 'GenerationMixin' from
        'transformers.generation'`` because peptdeep>=1.4 left
        transformers unbounded and pip was resolving 4.50+, which moved
        the symbol.  Every v0.5.8.x predicted-library / rescore / MBR
        donor-injection job therefore silently fell back to Sage's
        in-silico library; v0.5.8.1 made the failure visible in
        run_manifest.json and v0.5.9 actually fixes it.
  P0.   Build-time fail-fast smoke test (``python -c 'from
        peptdeep.pretrained_models import ModelManager'``) added to the
        Dockerfile so a regression of this class never reaches GHCR
        again -- the image build itself now red-flags the ABI break
        instead of producing a broken ``:latest`` tag.

v0.5.8.1 changes (the "observability hotfix" release):
  P0.   When predicted_library is enabled but the AlphaPeptDeep import,
        model load or actual prediction silently fails (returning ``None``
        so the caller falls back to Sage's in-silico library),
        ``run_manifest.json`` now records *why* in three new fields:
           predicted_library.reason            (e.g. ``alphapeptdeep_import_failed: ...``)
           predicted_library.peptdeep_importable / peptdeep_detail   (live import self-check)
           predicted_library.peptdeep_build_status                   (cached at image build time)
           rescoring.reason / mbr.injection_reason                   (downstream consequences)
        This fixes the v0.5.8 production diagnosis blind spot where
        predicted_library: applied=false / mbr.n_psms_rescued=0 left
        the operator with no way to tell whether the cause was the
        config, the container or the input.  No behavioural change for
        successful runs.

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

__version__ = "0.5.9.2"
