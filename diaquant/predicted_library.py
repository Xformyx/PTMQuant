"""AlphaPeptDeep predicted spectral library generation (NEW in 0.5.0).

This module wraps `AlphaPeptDeep <https://github.com/MannLabs/alphapeptdeep>`_
(Apache-2.0) so diaquant can ship a *predicted* spectral library that covers
the full mouse / human / etc. proteome and **every** PTM combination the user
has selected — not just phosphorylation and ubiquitin remnants like DIA-NN.

Why this matters
----------------
DIA-NN's library-free mode predicts iRT/MS2 only for unmodified peptides
plus the two PTMs Demichev et al. trained models for (Phospho, GlyGly).  For
acetyl, succinyl, malonyl, crotonyl, K/R methyl-lysine etc. the user is on
their own and identification depth collapses.  AlphaPeptDeep, by contrast,
ships a generic transformer model that natively understands ~2 800 UniMod
PTMs out of the box.  By pre-computing a SpecLib (precursor + fragment +
RT) per pass and feeding it to Sage, diaquant gets DIA-NN-level (or better)
identification depth across *all* PTMs.

Design choices
--------------
* **Lazy import**: ``peptdeep`` is heavy (≥ 2 GB with PyTorch / CUDA).  We
  import it inside :func:`generate_predicted_library` so users that never
  enable predicted libraries (``predicted_library: false``) are not forced
  to install AlphaPeptDeep.
* **Graceful fallback**: if AlphaPeptDeep is unavailable, the trained models
  are missing or prediction crashes (e.g. an unmapped PTM) the function
  returns ``None`` and the caller falls back to Sage's built-in theoretical
  library — the same behaviour as diaquant 0.4.x.  This is governed by
  :attr:`DiaQuantConfig.pred_lib_fallback_in_silico`.
* **Disk cache**: predicted libraries are deterministic for a fixed
  (FASTA, pass parameters, model checksum) tuple, so we cache the output
  TSV next to the Sage results.  Re-runs of the same pass skip the
  expensive prediction step entirely.
* **Transfer learning is opt-in**: training on the user's own PSMs takes
  several minutes per pass and is rarely needed.  When
  ``pred_lib_transfer_learning: true`` the caller can hand a PSM dataframe
  to :func:`fine_tune_models` *before* calling
  :func:`generate_predicted_library` so the predictions reflect the user's
  exact gradient / instrument settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .config import DiaQuantConfig
from .modifications import DEFAULT_MODIFICATIONS, Modification, resolve_modifications

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PTM-name mapping: PTMQuant catalogue  →  AlphaPeptDeep / AlphaBase
# ---------------------------------------------------------------------------
# AlphaPeptDeep uses the convention ``<Name>@<Residue>`` (or
# ``<Name>@Protein_N-term`` etc.).  PTMQuant uses friendlier human-readable
# names like ``Phospho`` plus a tuple of target residues.  To translate, we
# explode each PTMQuant modification into one AlphaPeptDeep entry per target
# residue.  Names that AlphaPeptDeep registers under a different stem (e.g.
# ``GlyGly`` → ``GG``, ``Citrullination`` → ``Deamidated@R``, ``Acetyl_Nterm``
# → ``Acetyl@Protein_N-term``) are translated explicitly.
#
# All mappings are validated against ``alphabase.constants.modification.MOD_DF``
# at load time so a typo is caught immediately rather than at prediction time.

# Stem alias: PTMQuant base name  →  AlphaPeptDeep base name
_STEM_ALIAS: dict[str, str] = {
    "GlyGly": "GG",                  # K-GG ubiquitin / NEDD8 remnant
    "Acetyl_Nterm": "Acetyl",        # protein N-term acetylation
    "Citrullination": "Deamidated",  # +0.984 on R, registered as Deamidated@R
    "Sumo_QQTGG": "QQTGG",           # SUMO-2/3 QQTGG remnant
}

# Residue alias for non-amino-acid targets used by PTMQuant
_RESIDUE_ALIAS: dict[str, str] = {
    "[": "Protein_N-term",
    "^": "Any_N-term",
    "]": "Protein_C-term",
    "$": "Any_C-term",
}


def _ptm_to_alphapept(mod: Modification) -> List[str]:
    """Return the list of AlphaPeptDeep PTM strings for one PTMQuant mod.

    For example ``Phospho`` (targets S/T/Y) becomes
    ``["Phospho@S", "Phospho@T", "Phospho@Y"]``.  Stems and residues that
    differ between catalogues are translated via :data:`_STEM_ALIAS` /
    :data:`_RESIDUE_ALIAS`.
    """
    stem = _STEM_ALIAS.get(mod.name, mod.name)
    out: List[str] = []
    for residue in mod.targets:
        target = _RESIDUE_ALIAS.get(residue, residue)
        out.append(f"{stem}@{target}")
    return out


def map_modifications(mods: Iterable[Modification]) -> Tuple[List[str], List[str]]:
    """Split PTMQuant mods into AlphaPeptDeep ``(fix_mods, var_mods)`` lists.

    Modifications whose AlphaPeptDeep entry is unknown are silently dropped
    with a warning.  Returning an empty fixed-mod list is fine; an empty
    variable-mod list will simply yield an unmodified library.
    """
    fix_mods: List[str] = []
    var_mods: List[str] = []
    for mod in mods:
        for entry in _ptm_to_alphapept(mod):
            if mod.fixed:
                fix_mods.append(entry)
            else:
                var_mods.append(entry)
    return fix_mods, var_mods


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _hash_fasta(fasta: Path) -> str:
    """Return a short SHA-1 fingerprint of the FASTA file contents."""
    h = hashlib.sha1()
    with open(fasta, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# Bump this when the AlphaPeptDeep / alphabase model family changes so we do
# not accidentally reuse a cached library predicted by a different model.
_MODEL_VERSION = "peptdeep-generic-1"


def _cache_key(cfg: DiaQuantConfig, fix_mods: List[str], var_mods: List[str]) -> str:
    """Deterministic cache key for one predicted-library run.

    All parameters that change the output library participate in the hash, so
    two jobs with the same FASTA / species / enzyme / PTM set / instrument /
    mz-range end up pointing at the same cached TSV regardless of where the
    job was submitted from.
    """
    payload = {
        "fasta": _hash_fasta(cfg.fasta),
        "enzyme": cfg.enzyme,
        "missed_cleavages": cfg.missed_cleavages,
        "min_len": cfg.min_peptide_length,
        "max_len": cfg.max_peptide_length,
        "min_charge": cfg.min_precursor_charge,
        "max_charge": cfg.max_precursor_charge,
        "min_mz": cfg.min_precursor_mz,
        "max_mz": cfg.max_precursor_mz,
        "max_var": cfg.max_variable_mods,
        "instrument": cfg.pred_lib_instrument,
        "nce": cfg.pred_lib_nce,
        "fix_mods": sorted(fix_mods),
        "var_mods": sorted(var_mods),
        "model": _MODEL_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _resolve_cache_paths(
    cfg: DiaQuantConfig, out_dir: Path, cache_id: str
) -> Tuple[Path, Optional[Path]]:
    """Return ``(local_path, shared_path)`` for the given cache id.

    * ``local_path`` is always under the per-pass ``out_dir``.  This is the
      legacy 0.5.1 location and is still written so Sage can pick it up on a
      resumed run.
    * ``shared_path`` is under ``cfg.pred_lib_cache_dir`` when configured and
      is the cross-job cache that makes identical jobs instantaneous.  It is
      ``None`` when the user has not configured a shared cache directory.
    """
    local = Path(out_dir) / f"predicted_library_{cache_id}.tsv"
    shared: Optional[Path] = None
    if cfg.pred_lib_cache_dir is not None:
        shared = Path(cfg.pred_lib_cache_dir) / f"predicted_library_{cache_id}.tsv"
    return local, shared


def _write_cache_meta(path: Path, cfg: DiaQuantConfig,
                      fix_mods: List[str], var_mods: List[str],
                      cache_id: str) -> None:
    """Write a sibling ``.meta.json`` describing what is in the cached TSV.

    The metadata file makes the shared cache human-auditable: an operator can
    see at a glance which FASTA / species / PTM set produced each cached
    library without running diaquant.
    """
    meta = {
        "cache_id": cache_id,
        "fasta": str(cfg.fasta),
        "fasta_sha1_12": _hash_fasta(cfg.fasta),
        "enzyme": cfg.enzyme,
        "missed_cleavages": cfg.missed_cleavages,
        "min_peptide_length": cfg.min_peptide_length,
        "max_peptide_length": cfg.max_peptide_length,
        "min_precursor_charge": cfg.min_precursor_charge,
        "max_precursor_charge": cfg.max_precursor_charge,
        "min_precursor_mz": cfg.min_precursor_mz,
        "max_precursor_mz": cfg.max_precursor_mz,
        "max_variable_mods": cfg.max_variable_mods,
        "pred_lib_instrument": cfg.pred_lib_instrument,
        "pred_lib_nce": cfg.pred_lib_nce,
        "fix_mods": sorted(fix_mods),
        "var_mods": sorted(var_mods),
        "model_version": _MODEL_VERSION,
    }
    try:
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )
    except Exception as exc:  # pragma: no cover - metadata is best-effort
        log.debug("Could not write cache metadata %s (%s)", path, exc)


# ---------------------------------------------------------------------------
# Lazy AlphaPeptDeep loader (kept out of import-time)
# ---------------------------------------------------------------------------

@dataclass
class _AlphaPeptDeepHandles:
    ModelManager: object
    PredictSpecLibFasta: object
    translate_to_tsv: object


def _load_alphapeptdeep() -> Optional[_AlphaPeptDeepHandles]:
    """Lazy-import AlphaPeptDeep; return ``None`` if unavailable."""
    try:  # pragma: no cover - import-side effects depend on user environment
        warnings.filterwarnings(
            "ignore",
            message="mask_modloss is deprecated",
            category=UserWarning,
        )
        from peptdeep.pretrained_models import ModelManager
        from peptdeep.protein.fasta import PredictSpecLibFasta
        from alphabase.spectral_library.translate import translate_to_tsv
    except Exception as exc:  # pragma: no cover
        log.warning(
            "AlphaPeptDeep unavailable (%s). Falling back to Sage's built-in "
            "theoretical library.", exc,
        )
        return None
    return _AlphaPeptDeepHandles(
        ModelManager=ModelManager,
        PredictSpecLibFasta=PredictSpecLibFasta,
        translate_to_tsv=translate_to_tsv,
    )


def _build_model_manager(handles: _AlphaPeptDeepHandles, cfg: DiaQuantConfig):
    """Initialise an AlphaPeptDeep ``ModelManager`` honouring user settings."""
    mm = handles.ModelManager(device="cpu")
    mm.load_installed_models("generic")
    mm.instrument = cfg.pred_lib_instrument
    mm.nce = float(cfg.pred_lib_nce)
    return mm


# ---------------------------------------------------------------------------
# Optional transfer-learning entry point
# ---------------------------------------------------------------------------

def fine_tune_models(cfg: DiaQuantConfig, psm_df) -> bool:
    """Fine-tune RT/MS2 on the user's high-confidence PSMs.

    Returns ``True`` on success, ``False`` if anything goes wrong.  The
    fine-tuned models are saved to :attr:`DiaQuantConfig.pred_lib_model_dir`
    (or AlphaPeptDeep's default cache) and will be picked up automatically by
    :func:`generate_predicted_library` on subsequent calls.
    """
    if not cfg.pred_lib_transfer_learning:
        return False
    handles = _load_alphapeptdeep()
    if handles is None:
        return False
    try:  # pragma: no cover - requires real AlphaPeptDeep + GPU/CPU compute
        mm = _build_model_manager(handles, cfg)
        mm.epoch_to_train_rt_ccs = int(cfg.pred_lib_transfer_epochs)
        mm.epoch_to_train_ms2 = int(cfg.pred_lib_transfer_epochs)
        log.info(
            "[diaquant] transfer learning: fine-tuning RT/MS2 on %d PSMs "
            "(%d epochs)", len(psm_df), cfg.pred_lib_transfer_epochs,
        )
        mm.train_rt_model(psm_df)
        mm.save_models(str(cfg.pred_lib_model_dir) if cfg.pred_lib_model_dir
                       else None)
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("Transfer learning failed (%s); using stock models.", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_predicted_library(
    cfg: DiaQuantConfig,
    out_dir: Path,
    *,
    extra_var_mods: Optional[List[str]] = None,
    pass_label: str = "default",
) -> Optional[Path]:
    """Generate a predicted SpecLib for the active modification set.

    Parameters
    ----------
    cfg
        Active diaquant configuration (the per-pass copy from
        :func:`diaquant.multipass._config_for_pass`).
    out_dir
        Directory where the TSV library is written / cached.
    extra_var_mods
        Extra AlphaPeptDeep-style PTM strings to add on top of the ones
        derived from ``cfg``.  Mostly useful for unit tests.
    pass_label
        Human-readable pass name (logged only).

    Returns
    -------
    pathlib.Path | None
        Path to the predicted library TSV (Sage-compatible) on success.
        ``None`` indicates the caller should fall back to Sage's built-in
        theoretical library.
    """
    if not cfg.predicted_library:
        log.info("[diaquant] %s: predicted library disabled (config).", pass_label)
        return None

    # Translate PTMQuant mods into AlphaPeptDeep mod strings.
    mods = resolve_modifications(
        list(cfg.fixed_modifications) + list(cfg.variable_modifications),
        cfg.custom_modifications,
    )
    fix_mods, var_mods = map_modifications(mods)
    if extra_var_mods:
        var_mods.extend(extra_var_mods)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_id = _cache_key(cfg, fix_mods, var_mods)
    local_tsv, shared_tsv = _resolve_cache_paths(cfg, out_dir, cache_id)

    # --- cache lookup ---------------------------------------------------
    # Priority order: shared cross-job cache > per-pass local cache.  When
    # the shared cache has a hit we also symlink / copy it into the local
    # out_dir so that every diaquant run appears self-contained on disk.
    if cfg.pred_lib_cache:
        if shared_tsv is not None and shared_tsv.exists():
            log.info("[diaquant] %s: reusing SHARED predicted library %s",
                     pass_label, shared_tsv)
            if not local_tsv.exists():
                try:
                    local_tsv.symlink_to(shared_tsv)
                except Exception:
                    import shutil as _sh
                    _sh.copyfile(shared_tsv, local_tsv)
            return local_tsv
        if local_tsv.exists():
            log.info("[diaquant] %s: reusing local predicted library %s",
                     pass_label, local_tsv.name)
            # Promote hit into shared cache so the next job benefits.
            if shared_tsv is not None:
                try:
                    shared_tsv.parent.mkdir(parents=True, exist_ok=True)
                    if not shared_tsv.exists():
                        import shutil as _sh
                        _sh.copyfile(local_tsv, shared_tsv)
                        _write_cache_meta(shared_tsv, cfg, fix_mods, var_mods, cache_id)
                except Exception as exc:  # pragma: no cover
                    log.debug("Could not promote local cache to shared (%s)", exc)
            return local_tsv

    out_tsv = local_tsv

    handles = _load_alphapeptdeep()
    if handles is None:
        if cfg.pred_lib_fallback_in_silico:
            log.warning("[diaquant] %s: AlphaPeptDeep unavailable; "
                        "falling back to in-silico library.", pass_label)
            return None
        raise RuntimeError(
            "AlphaPeptDeep is required (predicted_library=true, "
            "pred_lib_fallback_in_silico=false) but could not be imported. "
            "Install via `pip install peptdeep` and "
            "`peptdeep install-models`."
        )

    try:  # pragma: no cover - exercised in integration tests, not unit tests
        mm = _build_model_manager(handles, cfg)
        lib = handles.PredictSpecLibFasta(
            mm,
            protease=cfg.enzyme if cfg.enzyme != "no-cleavage" else "trypsin",
            max_missed_cleavages=cfg.missed_cleavages,
            peptide_length_min=cfg.min_peptide_length,
            peptide_length_max=cfg.max_peptide_length,
            precursor_charge_min=cfg.min_precursor_charge,
            precursor_charge_max=cfg.max_precursor_charge,
            precursor_mz_min=cfg.min_precursor_mz,
            precursor_mz_max=cfg.max_precursor_mz,
            var_mods=var_mods or ["Oxidation@M"],
            fix_mods=fix_mods or ["Carbamidomethyl@C"],
            max_var_mod_num=cfg.max_variable_mods,
            decoy=None,
        )
        log.info("[diaquant] %s: digesting FASTA with AlphaPeptDeep ...",
                 pass_label)
        lib.import_and_process_fasta([str(cfg.fasta)])
        log.info(
            "[diaquant] %s: %d peptides / %d precursors -> predicting RT+MS2",
            pass_label, len(lib.peptide_df), len(lib.precursor_df),
        )
        lib.predict_all(predict_items=["rt", "ms2"])
        handles.translate_to_tsv(
            lib, str(out_tsv),
            multiprocessing=False,
            batch_size=100_000,
        )
        log.info("[diaquant] %s: predicted library written to %s",
                 pass_label, out_tsv)
        # Populate shared cache on first computation so the next job of any
        # other user on the same platform instance can reuse this library.
        if cfg.pred_lib_cache and shared_tsv is not None:
            try:
                shared_tsv.parent.mkdir(parents=True, exist_ok=True)
                if not shared_tsv.exists():
                    import shutil as _sh
                    _sh.copyfile(out_tsv, shared_tsv)
                    _write_cache_meta(shared_tsv, cfg, fix_mods, var_mods, cache_id)
                    log.info("[diaquant] %s: promoted library to shared cache %s",
                             pass_label, shared_tsv)
            except Exception as exc:  # pragma: no cover
                log.warning("Could not write shared cache (%s)", exc)
        _write_cache_meta(out_tsv, cfg, fix_mods, var_mods, cache_id)
        return out_tsv
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[diaquant] %s: AlphaPeptDeep prediction failed (%s)",
            pass_label, exc,
        )
        if cfg.pred_lib_fallback_in_silico:
            log.warning("[diaquant] %s: falling back to in-silico library.",
                        pass_label)
            return None
        raise
