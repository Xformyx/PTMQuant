"""diaquant.alphadia_runner — invoke the AlphaDIA search engine.

This module is the v0.6.x replacement for ``diaquant.sage_runner``.

Why we replaced Sage
--------------------
Sage was designed around an *in-silico* protein digest as its primary
search input; the user-facing JSON config has no first-class field for an
external spectral library.  In PTMQuant v0.5.x we generated a 44.5 GB
AlphaPeptDeep predicted library that Sage simply ignored at search time
(it was used only as post-hoc rescoring fuel and as an MBR donor pool).
The KBSI-46847 mouse phospho-DIA diagnosis showed this cost us roughly
two-thirds of the recall DIA-NN was achieving on the same data.

AlphaDIA, in contrast, is built around the predicted library: the
library *is* the search.  Quoting Wallmann et al. 2025
(https://doi.org/10.1038/s41587-025-02845-z):

    "Through integration with the transformer models of alphaPeptDeep,
     alphaDIA closes the loop between spectral library prediction and
     DIA search."

That is exactly the loop PTMQuant has been trying to close since v0.5.0.
Because both AlphaDIA and AlphaPeptDeep ship from the MannLabs alphaX
ecosystem, the predicted library produced by ``predicted_library.py``
plugs into ``alphadia`` without any conversion.

Phase 1 scope
-------------
Skeleton — ``probe_alphadia`` + thin subprocess wrapper, transport-only
config keys (raw paths, library path, output dir, basic mass tolerances).

Phase 2 scope (THIS FILE)
-------------------------
PTM-aware ``build_alphadia_config``:

  * Translates each :class:`~diaquant.ptm_profiles.PassProfile` into the
    AlphaDIA YAML keys ``library_prediction.{variable_modifications,
    fixed_modifications, max_var_mod_num, missed_cleavages, precursor_len,
    precursor_charge, peptdeep_model_type}``.
  * Maps PTMQuant's per-pass ``peptide_fdr`` (0.01 / 0.05) onto AlphaDIA's
    single ``fdr.fdr`` cutoff.
  * Auto-selects the specialised PeptDeep model (``phospho`` for the
    phospho pass, ``digly`` for the K-GG ubiquitin pass, ``generic``
    otherwise) so that the in-silico prediction tier inside AlphaDIA
    benefits from the same family-specific models we already use in
    ``predicted_library.py``.
  * Emits the special form ``Acetyl@Protein_N-term`` for what PTMQuant
    historically called ``Acetyl_Nterm`` (target ``"["``); other PTMs
    expand on every target residue (e.g. ``Phospho`` → ``Phospho@S;
    Phospho@T;Phospho@Y``).

Phase 3 will hand the AlphaDIA precursor TSV to ``parse_alphadia.py`` and
on to directLFQ; Phase 4 wires up phospho-pass specifics; Phase 5
evaluates AlphaDIA's native MBR.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from diaquant.config import DiaQuantConfig
    from diaquant.ptm_profiles import PassProfile

log = logging.getLogger(__name__)

ALPHADIA_BIN = os.environ.get("PTMQUANT_ALPHADIA_BIN", "alphadia")


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaDIAProbe:
    """Result of ``probe_alphadia()``: tells the pipeline whether the engine
    is installed, and if not, *why* (so ``run_manifest.json`` can record
    the reason without falling back silently)."""

    available: bool
    binary: str
    version: Optional[str]
    reason: Optional[str]

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "binary": self.binary,
            "version": self.version,
            "reason": self.reason,
        }


def probe_alphadia(binary: str = ALPHADIA_BIN) -> AlphaDIAProbe:
    """Locate the ``alphadia`` CLI and capture its version string.

    Mirrors the diagnostic style of ``v0.5.8.1``'s ``predicted_library``
    probe so PTM-platform's UI can show *why* the engine is missing
    instead of just ``applied=false``.
    """
    resolved = shutil.which(binary)
    if not resolved:
        return AlphaDIAProbe(
            available=False,
            binary=binary,
            version=None,
            reason=f"alphadia binary not found on PATH (looked for {binary!r}); "
                   "rebuild the image with v0.6.0+ or install alphadia[stable]",
        )
    try:
        out = subprocess.check_output(
            [resolved, "--version"],
            stderr=subprocess.STDOUT,
            timeout=15,
            text=True,
        ).strip()
    except subprocess.SubprocessError as exc:
        return AlphaDIAProbe(
            available=False,
            binary=resolved,
            version=None,
            reason=f"alphadia --version failed: {exc!r}",
        )
    # AlphaDIA prints something like "alphadia 2.1.1"; tolerate any prefix
    version = out.splitlines()[-1].strip().split()[-1] if out else None
    return AlphaDIAProbe(
        available=True,
        binary=resolved,
        version=version,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Phase 2: PTM-aware config builder helpers
# ---------------------------------------------------------------------------

#: Translation table for residue tokens that PTMQuant uses internally vs.
#: AlphaDIA / alphabase modification.tsv conventions.  We deliberately do
#: NOT mutate ``DEFAULT_MODIFICATIONS`` in ``modifications.py`` — keeping
#: this translation local means the Sage-era code path stays bit-identical
#: while the AlphaDIA path emits the alphabase-compatible form.
#:
#: ``"["``  is PTMQuant's historic shorthand for the **protein** N-term
#:          (used by the built-in ``Acetyl_Nterm`` modification).
#: ``"]"``  is PTMQuant's historic shorthand for the **protein** C-term
#:          (currently unused by any built-in but reserved).
_RESIDUE_TO_ALPHADIA = {
    "[": "Protein_N-term",
    "]": "Protein_C-term",
}

#: PTM-pass name → specialised PeptDeep model.  AlphaDIA 2.x ships three
#: models out of the box (``generic``, ``phospho``, ``digly``).  Picking
#: the right one materially improves library quality for that PTM family.
#:
#: We key off ``PassProfile.name`` rather than the modification list to
#: keep the policy explicit and easy for users to override in YAML if
#: they extend ``PASS_PROFILES`` with a custom pass.
_PASS_TO_PEPTDEEP_MODEL = {
    "phospho": "phospho",
    "ubiquitin": "digly",
}


def _format_alphadia_modification(mod_name: str, residue: str) -> str:
    """Format a single ``Modification@Residue`` token for AlphaDIA.

    AlphaDIA expects either a one-letter residue (``S``, ``T``, ``K`` …)
    or one of the special tokens ``Protein_N-term`` / ``Protein_C-term``
    / ``Any_N-term`` / ``Any_C-term``.  We translate PTMQuant's historic
    bracket shorthand here.
    """
    target = _RESIDUE_TO_ALPHADIA.get(residue, residue)
    # PTMQuant stores ``Acetyl_Nterm`` as a separate Modification with
    # name ``Acetyl_Nterm`` and target ``[``.  AlphaDIA does not have a
    # ``_Nterm`` suffix in alphabase; the canonical token is just
    # ``Acetyl@Protein_N-term``.  Strip the ``_Nterm`` suffix so the
    # alphabase lookup succeeds.
    canonical_name = mod_name[:-6] if mod_name.endswith("_Nterm") else mod_name
    return f"{canonical_name}@{target}"


def _expand_modifications_to_alphadia(
    mod_names: Iterable[str],
    fixed: bool,
) -> List[str]:
    """Resolve PTMQuant built-in modification names → AlphaDIA tokens.

    Each modification expands across all of its ``targets`` so a single
    PTMQuant entry like ``Phospho`` (targets ``("S","T","Y")``) becomes
    three AlphaDIA tokens (``Phospho@S``, ``Phospho@T``, ``Phospho@Y``).

    ``fixed`` filters the input list:
      * ``True``  — keep only modifications declared with ``fixed=True``.
      * ``False`` — keep only variable modifications.
    """
    # Lazy import to avoid a circular dep with diaquant.config / pytest
    # collection at module load time.
    from diaquant.modifications import DEFAULT_MODIFICATIONS

    out: List[str] = []
    for name in mod_names or []:
        mod = DEFAULT_MODIFICATIONS.get(name)
        if mod is None:
            # Unknown built-in: skip silently (the same name will already
            # have been validated by ``resolve_modifications`` in the Sage
            # path; we don't want to double-fail the AlphaDIA path).
            continue
        if mod.fixed != fixed:
            continue
        for residue in mod.targets:
            out.append(_format_alphadia_modification(mod.name, residue))
    return out


def _peptdeep_model_type_for(pass_profile: Optional["PassProfile"]) -> str:
    """Pick the specialised PeptDeep model AlphaDIA should use for this pass.

    Falls back to ``"generic"`` when the pass name is unknown or no pass
    profile was supplied (single-pass Sage-era runs).
    """
    if pass_profile is None:
        return "generic"
    return _PASS_TO_PEPTDEEP_MODEL.get(pass_profile.name, "generic")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _alphadia_config_and_audit(
    cfg: "DiaQuantConfig",
    pass_profile: Optional["PassProfile"] = None,
    library_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[dict, dict]:
    """Build *(alphadia_config_dict, ptmquant_audit_dict)*.

    The first dict is AlphaDIA-legal only (no ``_*`` PTMQuant extensions);
    the second is written to ``ptmquant_alphadia_meta.json`` so
    run-manifest consumers can still audit pass / FDR / library mode.
    """
    out_dir = Path(output_dir) if output_dir else Path(cfg.output_dir) / "alphadia"
    raw_paths: list[str] = [str(Path(p).resolve()) for p in cfg.mzml_files]

    # ---- Resolve modifications --------------------------------------------
    # Prefer the pass-specific list; if none, fall back to the global
    # DiaQuantConfig.variable_modifications.
    if pass_profile is not None:
        var_names: List[str] = list(pass_profile.variable_modifications or [])
    else:
        var_names = list(getattr(cfg, "variable_modifications", []) or [])
    fixed_names: List[str] = list(getattr(cfg, "fixed_modifications", []) or [])

    var_tokens = _expand_modifications_to_alphadia(var_names, fixed=False)
    fixed_tokens = _expand_modifications_to_alphadia(fixed_names, fixed=True)
    # Some PTMQuant fixed mods (notably ``Carbamidomethyl``) were resolved
    # via the global ``fixed_modifications`` list above, but the *variable*
    # list might also contain the legacy ``Acetyl_Nterm`` entry that is
    # variable in PTMQuant.  ``_expand_modifications_to_alphadia`` already
    # filters by the ``fixed`` flag inside the Modification dataclass, so
    # nothing extra is needed here.

    # ---- Resolve numeric overrides (pass beats global) --------------------
    def _pick(field_name: str, default):
        if pass_profile is not None:
            v = getattr(pass_profile, field_name, None)
            if v is not None:
                return v
        return getattr(cfg, field_name, default)

    missed_cleavages = _pick("missed_cleavages", 2)
    max_var_mod_num = _pick("max_variable_mods", 2)
    min_peptide_length = _pick("min_peptide_length", 7)
    max_peptide_length = _pick("max_peptide_length", 30)
    max_precursor_charge = _pick("max_precursor_charge", 4)
    min_precursor_charge = int(getattr(cfg, "min_precursor_charge", 2))

    # ---- FDR --------------------------------------------------------------
    if pass_profile is not None and pass_profile.peptide_fdr is not None:
        fdr_cutoff = float(pass_profile.peptide_fdr)
    else:
        fdr_cutoff = float(getattr(cfg, "peptide_fdr", 0.01))

    # ---- PeptDeep model selection ----------------------------------------
    peptdeep_model_type = _peptdeep_model_type_for(pass_profile)

    # ---- Library mode flag -----------------------------------------------
    # If we have a pre-computed library, disable in-engine prediction so
    # AlphaDIA loads the library as-is.  Otherwise let AlphaDIA predict
    # from the FASTA using the (possibly specialised) PeptDeep model above.
    library_prediction_enabled = library_path is None

    config: dict = {
        # ---- I/O ----------------------------------------------------------
        "raw_paths": raw_paths,
        "output_directory": str(out_dir.resolve()),
        # NB: AlphaDIA's canonical key is ``library_path``; ``null`` means
        # "predict from FASTA at runtime".
        "library_path": str(Path(library_path).resolve()) if library_path else None,
        # AlphaDIA's canonical key is ``fasta_paths`` (not ``fasta_list``;
        # we corrected this from Phase 1's draft after inspecting
        # constants/default.yaml @ v2.1.1).
        "fasta_paths": [str(Path(cfg.fasta).resolve())] if cfg.fasta else [],

        # ---- Library prediction (PTM-aware) -------------------------------
        "library_prediction": {
            "enabled": library_prediction_enabled,
            "enzyme": getattr(cfg, "enzyme", "trypsin"),
            "fixed_modifications": ";".join(fixed_tokens) if fixed_tokens else "",
            "variable_modifications": ";".join(var_tokens) if var_tokens else "",
            "max_var_mod_num": int(max_var_mod_num),
            "missed_cleavages": int(missed_cleavages),
            "precursor_len": [int(min_peptide_length), int(max_peptide_length)],
            "precursor_charge": [int(min_precursor_charge), int(max_precursor_charge)],
            "precursor_mz": [
                float(getattr(cfg, "min_precursor_mz", 400.0)),
                float(getattr(cfg, "max_precursor_mz", 1200.0)),
            ],
            "fragment_mz": [
                float(getattr(cfg, "min_fragment_mz", 200.0)),
                float(getattr(cfg, "max_fragment_mz", 2000.0)),
            ],
            "peptdeep_model_type": peptdeep_model_type,
        },

        # ---- Search tolerances --------------------------------------------
        "search": {
            # AlphaDIA's canonical keys (NOT the Phase 1 placeholders):
            "target_ms1_tolerance": float(getattr(cfg, "precursor_tol_ppm", 10.0)),
            "target_ms2_tolerance": float(
                # Per-pass override wins (e.g. oglcnac pass relaxes to 15.0)
                pass_profile.fragment_tol_ppm
                if pass_profile is not None and pass_profile.fragment_tol_ppm is not None
                else getattr(cfg, "fragment_tol_ppm", 20.0)
            ),
        },

        # ---- FDR ----------------------------------------------------------
        "fdr": {
            "fdr": fdr_cutoff,
            # We default to 'genes' for protein inference because PTMQuant
            # ships a Mygene-based gene-level roll-up downstream and the
            # KBSI report tables key off Genes.  Users can override via
            # custom_passes if they want strict UniProt-accession grouping.
            "group_level": "genes",
            # 'heuristic' matches AlphaDIA's default and is the recommended
            # mode for label-free proteomics per Wallmann et al. 2025.
            "inference_strategy": "heuristic",
        },

    }

    audit: dict = {
        "pass_name": getattr(pass_profile, "name", None),
        "pass_is_whole_proteome": bool(
            getattr(pass_profile, "is_whole_proteome", False)
        ) if pass_profile is not None else None,
        "engine": "alphadia",
        "engine_phase": "v0.6.0a3-phase2",
        "ptmquant_var_mods_resolved": var_tokens,
        "ptmquant_fixed_mods_resolved": fixed_tokens,
        "ptmquant_fdr_source": (
            "pass_profile.peptide_fdr"
            if pass_profile is not None and pass_profile.peptide_fdr is not None
            else "cfg.peptide_fdr"
        ),
        "ptmquant_peptdeep_model_type": peptdeep_model_type,
        "ptmquant_library_mode": (
            "precomputed" if library_path else "in_engine_prediction"
        ),
    }
    # AlphaDIA 2.x treats the instance YAML as layered updates onto defaults;
    # arbitrary top-level keys (e.g. ``_ptmquant``) raise CONFIG_ERROR.  Keep
    # PTMQuant-only metadata in ``ptmquant_alphadia_meta.json`` beside the YAML.
    return config, audit


def build_alphadia_config(
    cfg: "DiaQuantConfig",
    pass_profile: Optional["PassProfile"] = None,
    library_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Return the AlphaDIA subprocess config dict **only** (no PTMQuant keys).

    Audit metadata formerly under ``_ptmquant`` is written next to the YAML by
    :func:`run_alphadia` as ``ptmquant_alphadia_meta.json``.
    """
    alphadia_payload, _audit = _alphadia_config_and_audit(
        cfg,
        pass_profile=pass_profile,
        library_path=library_path,
        output_dir=output_dir,
    )
    return alphadia_payload


# ---------------------------------------------------------------------------
# Subprocess invoker (Phase 1: thin wrapper)
# ---------------------------------------------------------------------------

class AlphaDIAError(RuntimeError):
    """Raised when the AlphaDIA subprocess exits non-zero."""


def run_alphadia(
    cfg: "DiaQuantConfig",
    pass_profile: Optional["PassProfile"] = None,
    library_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    extra_args: Sequence[str] = (),
    binary: str = ALPHADIA_BIN,
) -> Path:
    """Build the AlphaDIA config, write it to ``<output_dir>/config.yaml``,
    invoke the ``alphadia`` CLI, and return the resolved output directory.

    Phase 1 contract (unchanged in Phase 2):
      * Hard-fails with :class:`AlphaDIAError` if ``alphadia`` is missing or
        exits non-zero.  No silent fallback to Sage — that policy mirrors
        the ``pred_lib_fallback_in_silico=False`` default introduced in
        v0.5.9.1: "better to surface the failure than to ship a degraded
        result."
      * Emits a JSON-formatted progress line at start and at completion
        so PTM-platform's UI can parse it (compatible with the v0.5.10
        ``run_metrics_*.jsonl`` consumer).
    """
    import time

    out_dir = Path(output_dir) if output_dir else Path(cfg.output_dir) / "alphadia"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_alphadia(binary)
    if not probe.available:
        raise AlphaDIAError(
            f"alphadia engine unavailable: {probe.reason}. "
            "Rebuild the v0.6.0+ image or install with `pip install alphadia[stable]`."
        )

    cfg_dict, audit = _alphadia_config_and_audit(
        cfg,
        pass_profile=pass_profile,
        library_path=library_path,
        output_dir=out_dir,
    )

    # AlphaDIA accepts YAML; we write JSON-as-YAML because JSON is a strict
    # YAML subset, which keeps this module free of a hard PyYAML import.
    config_path = out_dir / "alphadia_config.yaml"
    config_path.write_text(json.dumps(cfg_dict, indent=2, sort_keys=False))
    meta_path = out_dir / "ptmquant_alphadia_meta.json"
    meta_path.write_text(json.dumps(audit, indent=2, sort_keys=False), encoding="utf-8")

    cmd: list[str] = [probe.binary, "--config", str(config_path), *map(str, extra_args)]

    log.info(json.dumps({
        "event": "alphadia_start",
        "pass": audit.get("pass_name"),
        "binary": probe.binary,
        "version": probe.version,
        "config": str(config_path),
        "n_raw": len(cfg_dict["raw_paths"]),
        "library_path": cfg_dict["library_path"],
        "var_mods": cfg_dict["library_prediction"]["variable_modifications"],
        "fdr": cfg_dict["fdr"]["fdr"],
        "peptdeep_model_type": cfg_dict["library_prediction"]["peptdeep_model_type"],
    }))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(out_dir),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise AlphaDIAError(
            f"alphadia exited with code {proc.returncode} after {elapsed:.1f}s "
            f"(config: {config_path})"
        )

    log.info(json.dumps({
        "event": "alphadia_done",
        "pass": audit.get("pass_name"),
        "elapsed_sec": round(elapsed, 2),
        "output_dir": str(out_dir),
    }))
    return out_dir


__all__ = [
    "AlphaDIAError",
    "AlphaDIAProbe",
    "ALPHADIA_BIN",
    "build_alphadia_config",
    "probe_alphadia",
    "run_alphadia",
]
