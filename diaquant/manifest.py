"""v0.5.4: write a per-run ``run_manifest.json`` alongside the output matrices.

The manifest is the single source of truth that lets the user verify, without
re-reading the container, exactly which diaquant version ran, whether the
AlphaPeptDeep predicted library was actually produced and consumed, which
cache file was reused, which peptide FDR was applied, and how many PSMs were
rescored.  It also copies ``config.yaml`` and the effective
``predicted_library_*.tsv`` (or a symlink when the library lives in the
shared cache) into the output directory for the same observability reason.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    return str(obj)


def _copy_predicted_libraries(cfg,
                              out_dir: Path,
                              library_paths: List[Path]) -> List[Dict[str, Any]]:
    """Copy (or symlink) every predicted library file into out_dir.

    Returns a list of ``{"source", "copy", "size_bytes", "n_rows"}`` entries
    the manifest can embed.  Rows are counted cheaply by scanning the TSV.
    """
    entries: List[Dict[str, Any]] = []
    pred_dir = out_dir / "predicted_libraries"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for src in library_paths:
        if not src or not src.exists():
            continue
        dst = pred_dir / src.name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # Prefer a symlink when src lives on the same filesystem; fall back
            # to a real copy otherwise (shared Docker volume crossing devices).
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
        except Exception as exc:
            logger.warning("failed to mirror predicted library %s: %s", src, exc)
            continue
        size = src.stat().st_size
        # quick row count (header + rows)
        try:
            with src.open() as fh:
                n_rows = sum(1 for _ in fh) - 1
        except Exception:
            n_rows = -1
        entries.append({
            "source": str(src),
            "copy": str(dst),
            "size_bytes": size,
            "n_precursors": max(n_rows, 0),
            "is_symlink": dst.is_symlink(),
        })
    return entries


def write_run_manifest(
    cfg,
    out_dir: Path,
    *,
    diaquant_version: str,
    config_yaml_src: Optional[Path],
    library_paths: Optional[List[Path]] = None,
    n_psms_raw: Optional[int] = None,
    n_psms_rescored: Optional[int] = None,
    n_psms_after_fdr: Optional[int] = None,
    n_psms_mbr: Optional[int] = None,
    pr_rows: Optional[int] = None,
    pg_rows: Optional[int] = None,
    site_rows: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist run_manifest.json + config.yaml copy + predicted_library copies.

    Parameters
    ----------
    cfg
        The active ``DiaQuantConfig`` instance.  Serialised via ``asdict``.
    out_dir
        Run output directory.  Must already exist.
    diaquant_version
        ``diaquant.__version__`` — captured here so the user sees which git
        tag actually ran inside the container.
    config_yaml_src
        Path to the original config.yaml the user launched the run with.
        The file is copied verbatim into ``out_dir/config.yaml`` (skipped if
        the destination and source resolve to the same file).
    library_paths
        List of predicted-library TSVs written by :mod:`.predicted_library`.
        Each file is mirrored into ``out_dir/predicted_libraries/``.
    n_psms_raw, n_psms_rescored, n_psms_after_fdr
        Diagnostic counters.  ``rescored`` is non-None only when
        ``rescore_with_prediction`` is True *and* the predicted library was
        successfully joined; this lets the user distinguish "predicted library
        disabled" from "predicted library enabled but zero joins".
    pr_rows, pg_rows, site_rows
        Row counts of the three matrices actually written.
    extra
        Free-form dictionary merged into the manifest (e.g. RT alignment
        RMSE, batching info).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- copy config.yaml so the user can always inspect what ran ----
    copied_yaml: Optional[str] = None
    if config_yaml_src and Path(config_yaml_src).exists():
        dst = out_dir / "config.yaml"
        try:
            src_abs = Path(config_yaml_src).resolve()
            dst_abs = dst.resolve() if dst.exists() else dst.absolute()
            if src_abs != dst_abs:
                shutil.copy2(src_abs, dst)
            copied_yaml = str(dst)
        except Exception as exc:
            logger.warning("failed to copy config.yaml: %s", exc)

    # ---- mirror predicted libraries ----
    # v0.5.5: if the caller did not explicitly thread paths through, look for
    # predicted_library_*.tsv next to the output dir (pass_phospho/, etc.) and
    # in the configured cache dir; this makes multi-pass outputs observable.
    paths: List[Path] = [Path(p) for p in (library_paths or []) if p]
    # Only auto-discover when the caller did not pass a library_paths argument
    # at all (None).  An explicit empty list means "we already checked, zero
    # libraries were produced" and must be respected.
    if library_paths is None and not paths:
        search_roots = [out_dir, out_dir.parent]
        cache_dir = getattr(cfg, "pred_lib_cache_dir", None)
        if cache_dir:
            search_roots.append(Path(cache_dir))
        seen: set = set()
        for root in search_roots:
            if not root or not Path(root).exists():
                continue
            for p in Path(root).rglob("predicted_library_*.tsv"):
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(p)
    lib_entries = _copy_predicted_libraries(cfg, out_dir, paths)

    predicted_library_applied = (
        getattr(cfg, "predicted_library", False)
        and len(lib_entries) > 0
    )
    rescore_applied = (
        bool(getattr(cfg, "rescore_with_prediction", False))
        and predicted_library_applied
        and (n_psms_rescored is None or n_psms_rescored > 0)
    )

    manifest: Dict[str, Any] = {
        "diaquant_version": diaquant_version,
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "config_yaml_copy": copied_yaml,
        "env": {
            "PTMQUANT_LIB_CACHE_DIR": os.environ.get("PTMQUANT_LIB_CACHE_DIR"),
            "HOSTNAME": os.environ.get("HOSTNAME"),
        },
        "effective_config": _json_safe(cfg),
        "predicted_library": {
            "enabled_in_config": bool(getattr(cfg, "predicted_library", False)),
            "applied": predicted_library_applied,
            "cache_dir": str(getattr(cfg, "pred_lib_cache_dir", "") or ""),
            "files": lib_entries,
        },
        "rescoring": {
            "enabled_in_config": bool(getattr(cfg, "rescore_with_prediction", False)),
            "applied": rescore_applied,
            "n_psms_raw": n_psms_raw,
            "n_psms_rescored": n_psms_rescored,
            "n_psms_after_fdr": n_psms_after_fdr,
        },
        "mbr": {
            "enabled_in_config": bool(getattr(cfg, "match_between_runs", False)),
            "n_psms_rescued": int(n_psms_mbr) if n_psms_mbr is not None else 0,
        },
        "matrices": {
            "pr_matrix_rows": pr_rows,
            "pg_matrix_rows": pg_rows,
            "ptm_site_matrix_rows": site_rows,
        },
    }
    if extra:
        manifest["extra"] = _json_safe(extra)

    path = out_dir / "run_manifest.json"
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2, default=_json_safe, sort_keys=False)
    logger.info("run_manifest written to %s", path)
    return path
