"""diaquant: open-source DIA proteomics quantification pipeline.

Modules:
- config:             parse YAML configuration
- enzymes:            9-entry protease catalog (NEW in 0.5.1)
- instruments:        4-entry Orbitrap preset catalog (NEW in 0.5.1)
- modifications:      UniMod / custom PTM definitions
- ptm_profiles:       PTM-family-specific Sage search profiles (multi-pass)
- sage_runner:        build Sage JSON config and invoke the binary
- alphadia_runner:    v0.6.0 Phase 1 — invoke the AlphaDIA CLI; will
                      replace sage_runner as the primary engine in v0.6.0
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

v0.6.0a3 changes ("AlphaDIA Phase 2 — PTM-aware config builder"):
  P1.   alphadia_runner.build_alphadia_config() now translates each
        diaquant.ptm_profiles.PassProfile into a fully populated
        AlphaDIA YAML dict.  Highlights:
          * Variable / fixed modifications expand from PTMQuant built-in
            names to AlphaDIA's `Name@Target` tokens, joined by `;`.
            `Acetyl_Nterm` is rewritten to `Acetyl@Protein_N-term` so
            alphabase recognises it; `Phospho` (targets S/T/Y) becomes
            `Phospho@S;Phospho@T;Phospho@Y`; multi-target K/R PTMs
            (Methyl, Dimethyl) expand symmetrically.
          * Pass-level overrides (missed_cleavages, max_variable_mods,
            min/max peptide length, max precursor charge, fragment
            tolerance) win over the global DiaQuantConfig defaults.
            None means "inherit".
          * Per-pass peptide_fdr (0.05 for PTM passes, 0.01 for the
            whole-proteome backbone) maps onto AlphaDIA's single
            fdr.fdr cutoff.  This preserves the v0.5.8 PTM-FDR policy
            ("5% peptide FDR for low-frequency PTM populations, 1% for
            the proteome backbone") inside the new engine.
          * Specialised PeptDeep model auto-selection: phospho pass
            ->`peptdeep_model_type: phospho`; ubiquitin pass -> `digly`;
            everything else -> `generic`.  This matches the family-
            specific models PTMQuant has been using inside
            predicted_library.py since v0.5.0.
          * library_path argument flips library_prediction.enabled to
            False so a v0.5.x AlphaPeptDeep TSV/HDF library is consumed
            as-is rather than re-predicted by the engine.
          * The audit block under _ptmquant in the emitted YAML records
            the resolved tokens, FDR source, library mode, and PeptDeep
            model choice for downstream run_manifest.json forensics.
  P1.   tests/test_alphadia_runner.py (new): 16 cases covering whole
        proteome / phospho / ubiquitin / acetyl_methyl / oglcnac
        passes, library-path toggle, no-pass fallback, and a
        parametrised sanity check that every built-in pass produces a
        JSON-serialisable, schema-complete AlphaDIA config dict.  All
        82 (66 smoke + 16 phase-2) tests pass on the v0.6.0a2 main env.

v0.6.0a2 changes ("AlphaDIA Phase 1 — isolated venv hotfix"):
  P0.   Reinstall AlphaDIA into an isolated venv at /opt/alphadia-venv
        instead of the main Python env.  v0.6.0a1's flat install
        upgraded torch 2.2.2+cpu -> 2.6.0+cu124 but left torchvision
        0.17.2 behind, producing
            RuntimeError: operator torchvision::nms does not exist
        at the next `from peptdeep.pretrained_models import ModelManager`
        smoke test, killing the Docker build.  The venv approach
        guarantees the v0.5.10 CPU-only stack (torch 2.2.2+cpu,
        peptdeep, transformers 4.47.0, numba 0.60.0, numpy<2) is left
        bit-identical, while the `alphadia` CLI is exposed via a
        symlink at /usr/local/bin/alphadia.
  P0.   Add explicit regression-guard smoke tests in the Dockerfile:
           python -c "import torch, transformers, numba, numpy"
           python -c "from peptdeep.pretrained_models import ModelManager"
        Both must pass AFTER the alphadia install or the build fails
        fast.  This guarantees v0.6.0+ images can never silently break
        the validated v0.5.10 quantification path.

v0.6.0a1 changes (the "AlphaDIA Phase 1 — Docker integration" alpha):
  P0.   Add `mono-runtime` + `libmono-system-data4.0-cil` to the base
        image.  These are required by `alpharaw` (the AlphaDIA backend)
        to read Thermo `.raw` files on Linux.  Without them, AlphaDIA
        falls back to mzML-only and cannot consume native Orbitrap
        output, which is exactly the data class PTMQuant targets.
  P0.   Install `alphadia[stable]` (Apache-2.0, MannLabs).  AlphaDIA is
        the v0.6.x replacement for Sage as the primary DIA search engine.
        It is library-driven: AlphaPeptDeep's predicted library finally
        becomes a first-class search input rather than a post-hoc
        rescoring afterthought.  See Wallmann et al., Nat Biotechnol
        (2025): "alphaDIA closes the loop between spectral library
        prediction and DIA search."
  P0.   Re-pin `transformers==4.47.0` + `numba==0.60.0` + `numpy<2`
        AFTER the alphadia install so its dependency tree cannot relax
        the v0.5.9 ABI guarantees.
  P0.   Build-time fail-fast smoke tests: `import alphadia` and
        `alphadia --help` must both succeed before the image is
        published to GHCR, mirroring the v0.5.9 peptdeep fail-fast
        policy.
  P1.   New module `diaquant.alphadia_runner` with `probe_alphadia()`,
        `build_alphadia_config()` and `run_alphadia()`.  Phase 1 ships
        only the skeleton (CLI invocation + diagnostic probe); Phase 2
        will populate the YAML config builder with PTM-aware variable
        mods, pass-specific FDR, and MBR settings; Phase 3 will hand the
        AlphaDIA precursor TSV to a new `parse_alphadia.py` and on into
        the existing directLFQ + dask quantification path that v0.5.9.2
        accelerated.
  P1.   New CLI command `diaquant probe-alphadia` emits a JSON probe
        suitable for PTM-platform's pre-flight check.  Exits non-zero
        when alphadia is unreachable, so it can be wired into a Docker
        HEALTHCHECK or CI gate.
  P2.   No behavioural change to the v0.5.10 Sage path — every existing
        knob (`pred_lib_chunk_size`, `quant_num_cores`,
        `mbr_inject_predicted_donors`, etc.) is preserved verbatim.
        v0.5.10 remains the production-recommended tag until v0.6.0
        promotes out of alpha.

v0.5.10 changes (the "pre-search filter + MBR OOM hardening" release):
  P0.   Pre-search filter applied before AlphaPeptDeep predicted-library
        construction prunes the candidate precursor pool by ~95%
        (whole-proteome, multi-PTM jobs).  This was the primary cause of
        the v0.5.9.x peak RSS ceiling on the Mac Studio (96 GB Docker
        Desktop) and added unnecessary latency to every downstream step.
        The filter retains decoys and FDR semantics; only candidates with
        no mass-window / charge / RT overlap to the observed MS1 envelopes
        are dropped before peptdeep is even called.  See commit a750c8d.
  P0.   MBR donor-injection OOM hardening (commits 4825dee + c9ec319 +
        c9fc342).  The donor table is now loaded with only the three
        columns the matcher needs from the (often >40 GB) predicted
        library; the predicted-donor injection path is disabled by
        default (operators can re-enable via
        ``mbr_inject_predicted_donors: true``); and ``psm_full`` is
        filtered in place instead of via a full copy before slicing.
        These three together remove the SIGKILL-on-MBR class of failure
        observed on KBSI-46847.
  P0.   Release AlphaPeptDeep / torch memory before spawning the Sage
        subprocess (commit 19da1e9).  Without this the parent Python
        process held ~30 GB of model weights + tensor caches alive while
        Sage itself was already running, doubling peak RSS unnecessarily
        and triggering the kernel OOM killer on borderline-sized hosts.
  P1.   Streaming library join + checkpointing + JSON progress
        (commit 00796d0).  The library merge no longer materialises the
        full join in memory; per-pass checkpoints let an interrupted run
        resume; and a new JSON-formatted progress stream feeds the
        platform UI without parsing free-form log lines.  Cache key was
        refined so semantically identical configs no longer rebuild the
        library.
  P1.   Per-run resource metrics logger (commits 9012b24 + ea9f218).
        ``run_metrics_<YYYYMMDD_HHMMSS>.jsonl`` records RSS, CPU%,
        wall-clock and per-step elapsed for every pipeline stage,
        enabling post-mortem performance analysis without rerunning.
  P2.   No new APIs; all v0.5.9.2 configuration knobs
        (``pred_lib_chunk_size``, ``quant_num_cores``,
        ``PTMQUANT_QUANT_CORES``, etc.) are preserved verbatim.

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

__version__ = "0.6.0a3"
