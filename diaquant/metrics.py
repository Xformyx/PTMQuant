"""Per-run resource metrics logger (v0.5.9.3).

Writes one JSON line per pipeline event to ``<output_dir>/run_metrics.jsonl``.
Each line captures a timestamp, stage name, event type and a resource snapshot
(host RAM, process RSS, CPU %).  The file accumulates across the whole run so
you can compare CPU/memory curves between different builds or configurations.

Usage
-----
::

    from .metrics import record_event, set_metrics_dir, flush_metrics

    set_metrics_dir(cfg.output_dir)       # call once at pipeline start
    record_event("pipeline", "start")
    # ... work ...
    record_event("library", "chunk_done", chunk=1, total=27)
    flush_metrics()                       # optional; file is flushed after each write

Format
------
Each line is a self-contained JSON object::

    {
        "ts":         1746356400.123,     # Unix timestamp (float)
        "iso":        "2026-05-04T...",   # ISO-8601 for human readers
        "elapsed_s":  42.1,               # seconds since set_metrics_dir() call
        "stage":      "library",
        "event":      "chunk_done",
        "chunk":      1,
        "total":      27,
        "cpu_pct":    493.2,              # host CPU % (all cores, psutil)
        "mem_used_gb":   55.3,            # host used RAM GB
        "mem_avail_gb":  16.7,            # host available RAM GB
        "mem_pct":       76.8,            # host RAM %
        "proc_rss_gb":   22.1             # this process RSS GB
    }

Comparison workflow
-------------------
After two runs (old vs new build) produce their own ``run_metrics.jsonl``
files, diff them with::

    python - <<'EOF'
    import json, pathlib, statistics
    for label, path in [("old", "old/run_metrics.jsonl"),
                        ("new", "new/run_metrics.jsonl")]:
        events = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l]
        lib_chunks = [e for e in events if e["stage"] == "library" and e["event"] == "chunk_done"]
        if lib_chunks:
            times = [e["elapsed_s"] for e in lib_chunks]
            mems  = [e["mem_used_gb"] for e in lib_chunks]
            print(f"{label}: {len(lib_chunks)} chunks  "
                  f"total_elapsed={times[-1]:.0f}s  "
                  f"mem_used max={max(mems):.1f}GB  avg={statistics.mean(mems):.1f}GB")
    EOF
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (set once per pipeline run)
# ---------------------------------------------------------------------------

_out_dir: Optional[Path] = None
_start_time: float = 0.0
_metrics_path: Optional[Path] = None


def set_metrics_dir(out_dir: Path) -> None:
    """Configure the output directory and reset the elapsed-time origin.

    Call this **once** at the very start of each pipeline run (before any
    ``record_event`` call).  Subsequent calls overwrite the previous file
    path and reset the elapsed timer.
    """
    global _out_dir, _start_time, _metrics_path
    _out_dir = Path(out_dir)
    _out_dir.mkdir(parents=True, exist_ok=True)
    _start_time = time.monotonic()
    _metrics_path = _out_dir / "run_metrics.jsonl"


def _resource_snapshot() -> dict:
    """Collect CPU % and memory from psutil; return safe defaults if unavailable."""
    snap: dict = {
        "cpu_pct":      None,
        "mem_used_gb":  None,
        "mem_avail_gb": None,
        "mem_pct":      None,
        "proc_rss_gb":  None,
    }
    try:
        import psutil  # type: ignore

        # Host-level memory (instantaneous, no blocking)
        vm = psutil.virtual_memory()
        snap["mem_used_gb"]  = round(vm.used  / (1024 ** 3), 2)
        snap["mem_avail_gb"] = round(vm.available / (1024 ** 3), 2)
        snap["mem_pct"]      = round(vm.percent, 1)

        # Host CPU % accumulated since last call (interval=None = non-blocking,
        # returns utilisation since the PREVIOUS call to cpu_percent on this process).
        # We use interval=0.1 only on the first call to avoid a zero return.
        snap["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)

        # Current process RSS (resident set size)
        proc = psutil.Process()
        snap["proc_rss_gb"] = round(proc.memory_info().rss / (1024 ** 3), 2)
    except Exception:
        pass
    return snap


def record_event(stage: str, event: str, **extra: Any) -> None:
    """Write one metrics line to ``run_metrics.jsonl``.

    Parameters
    ----------
    stage
        Logical pipeline stage, e.g. ``"pipeline"``, ``"library"``,
        ``"sage"``, ``"rescore"``, ``"rt_align"``, ``"mbr"``, ``"quant"``.
    event
        Event within the stage, e.g. ``"start"``, ``"end"``,
        ``"chunk_done"``, ``"cache_hit"``, ``"checkpoint_hit"``.
    **extra
        Any additional key-value pairs appended to the JSON line
        (e.g. ``chunk=1``, ``total=27``, ``pass_name="phospho"``).
    """
    if _metrics_path is None:
        # Metrics not yet initialised — silently drop.  This happens when
        # a unit test calls pipeline code without calling set_metrics_dir.
        return

    now = time.time()
    elapsed = time.monotonic() - _start_time if _start_time else 0.0

    row: dict = {
        "ts":        round(now, 3),
        "iso":       _iso(now),
        "elapsed_s": round(elapsed, 1),
        "stage":     stage,
        "event":     event,
    }
    row.update(extra)
    row.update(_resource_snapshot())

    try:
        with open(_metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:   # non-fatal: never crash the pipeline for metrics
        log.debug("metrics write failed: %s", exc)


def flush_metrics() -> Optional[Path]:
    """Return the path of the metrics file (None if not initialised)."""
    return _metrics_path


def _iso(ts: float) -> str:
    """Return ISO-8601 UTC string for a Unix timestamp."""
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
