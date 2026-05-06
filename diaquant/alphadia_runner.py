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
This file currently exposes only a *skeleton*: enough surface area for
``cli.py`` and ``multipass.py`` to import and stub-out the engine choice,
plus a thin ``subprocess`` wrapper around the ``alphadia`` CLI so that
the v0.6.0-alpha.1 image can be exercised end-to-end with a tiny test
fixture.  Phase 2 will fill in the YAML config builder for
PTM-pass-aware searches; Phase 3 will hand the AlphaDIA precursor TSV to
``parse_alphadia.py`` and on to directLFQ; Phase 4 wires up phospho-pass
specifics; Phase 5 evaluates AlphaDIA's native MBR.
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
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from diaquant.config import DiaQuantConfig
    from diaquant.ptm_profiles import PTMProfile

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
# Config builder (Phase 1: skeleton; Phase 2 will flesh out PTM mapping)
# ---------------------------------------------------------------------------

def build_alphadia_config(
    cfg: "DiaQuantConfig",
    pass_profile: Optional["PTMProfile"] = None,
    library_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Translate the PTMQuant ``DiaQuantConfig`` into an AlphaDIA YAML
    config (returned as a dict; the caller serialises it).

    Phase 1 emits only the *transport* keys (raw paths, library path,
    output directory, basic mass tolerances).  PTM-aware variable mods,
    pass-specific FDR, MBR toggles, and quant settings are added in
    Phase 2 once we wire ``alphadia_runner`` into ``multipass.py``.

    The returned mapping intentionally uses AlphaDIA's snake_case keys so
    that ``yaml.safe_dump`` of the result is a valid AlphaDIA config.
    """
    out_dir = Path(output_dir) if output_dir else Path(cfg.output_dir) / "alphadia"
    raw_paths: list[str] = [str(Path(p).resolve()) for p in cfg.mzml_files]

    config: dict = {
        # ---- I/O ----------------------------------------------------------
        "raw_paths": raw_paths,
        "output_directory": str(out_dir.resolve()),
        "library_path": str(Path(library_path).resolve()) if library_path else None,
        "fasta_list": [str(Path(cfg.fasta).resolve())] if cfg.fasta else [],
        # ---- Search tolerances (Phase 1: pass through PTMQuant defaults) -
        "search": {
            "precursor_mz_tolerance_ppm": float(getattr(cfg, "precursor_tol_ppm", 10.0)),
            "fragment_mz_tolerance_ppm": float(getattr(cfg, "fragment_tol_ppm", 20.0)),
            # AlphaDIA's library mode flag: True == fully predicted library
            # (our case); False would mean "empirical DDA-derived library".
            "library_prediction": False if library_path else True,
        },
        # ---- Pass metadata (recorded so run_manifest can audit which pass
        #      drove which AlphaDIA invocation) ------------------------
        "_ptmquant": {
            "pass_name": getattr(pass_profile, "name", None),
            "engine": "alphadia",
            "engine_phase": "v0.6.0-alpha.1-skeleton",
        },
    }
    return config


# ---------------------------------------------------------------------------
# Subprocess invoker (Phase 1: thin wrapper)
# ---------------------------------------------------------------------------

class AlphaDIAError(RuntimeError):
    """Raised when the AlphaDIA subprocess exits non-zero."""


def run_alphadia(
    cfg: "DiaQuantConfig",
    pass_profile: Optional["PTMProfile"] = None,
    library_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    extra_args: Sequence[str] = (),
    binary: str = ALPHADIA_BIN,
) -> Path:
    """Build the AlphaDIA config, write it to ``<output_dir>/config.yaml``,
    invoke the ``alphadia`` CLI, and return the resolved output directory.

    Phase 1 contract:
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

    # Lazy import to avoid a circular dep with diaquant.config at module load.
    out_dir = Path(output_dir) if output_dir else Path(cfg.output_dir) / "alphadia"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_alphadia(binary)
    if not probe.available:
        raise AlphaDIAError(
            f"alphadia engine unavailable: {probe.reason}. "
            "Rebuild the v0.6.0+ image or install with `pip install alphadia[stable]`."
        )

    cfg_dict = build_alphadia_config(
        cfg,
        pass_profile=pass_profile,
        library_path=library_path,
        output_dir=out_dir,
    )

    # AlphaDIA accepts YAML; we write JSON-as-YAML for Phase 1 to avoid a
    # hard dependency on PyYAML at this layer (PyYAML is already a
    # transitive dep, but JSON is a strict YAML subset and keeps the
    # skeleton minimal).
    config_path = out_dir / "alphadia_config.yaml"
    config_path.write_text(json.dumps(cfg_dict, indent=2, sort_keys=False))

    cmd: list[str] = [probe.binary, "--config", str(config_path), *map(str, extra_args)]

    log.info(json.dumps({
        "event": "alphadia_start",
        "pass": cfg_dict["_ptmquant"]["pass_name"],
        "binary": probe.binary,
        "version": probe.version,
        "config": str(config_path),
        "n_raw": len(cfg_dict["raw_paths"]),
        "library_path": cfg_dict["library_path"],
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
        "pass": cfg_dict["_ptmquant"]["pass_name"],
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
