#!/usr/bin/env python3
"""Verify a PTMQuant (diaquant) output directory.

Usage
-----
    python scripts/verify_ptmquant.py /path/to/output_dir [options]

Exit codes
----------
    0  All mandatory checks passed.
    1  One or more mandatory checks failed.
    2  The output directory is unusable (missing / empty / not a dir).

The verifier intentionally separates *mandatory* checks (which fail the
run) from *advisory* checks (which only emit warnings).  This is so that
CI pipelines can gate on exit code 0 while still getting a detailed
report when the pipeline technically succeeded but did not benefit from
v0.5.5 features (e.g. MBR skipped because only one run was processed).

Checks
------
Mandatory
  1. ``run_manifest.json`` exists and is valid JSON.
  2. ``diaquant_version`` >= the user-supplied ``--min-version`` (default
     reads the caller's own installed ``diaquant.__version__``).
  3. ``report.pr_matrix.tsv`` exists and has at least one data row.
  4. ``report.pg_matrix.tsv`` exists and has at least one data row.
  5. ``report.pg_matrix.tsv`` Genes column is populated in at least the
     requested fraction of rows (``--min-genes-frac``, default 0.9).
  6. ``report.pg_matrix.tsv`` row count is at least ``min_pg_ratio``
     (default 0.5) times the number of distinct accessions in
     ``report.pr_matrix.tsv``'s ``Protein.Group`` column.

Advisory (warnings only, never fail the run)
  A. ``predicted_library.applied`` is true when AlphaPeptDeep is present.
  B. ``rescoring.applied`` is true when rescoring was enabled in config.
  C. ``mbr.applied`` is true and ``mbr.n_rescued`` > 0 when MBR is
     enabled in config.
  D. ``report.ptm_site_matrix.tsv`` exists and has at least one data
     row *when the run included any PTM-aware pass*.

The script is deliberately dependency-light (stdlib only) so it can run
inside the image itself (``docker run --entrypoint python ptmquant:latest
/app/scripts/verify_ptmquant.py /output``) as well as on the host.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Terminal colours (fallback to plain text if we're not a tty).
# ---------------------------------------------------------------------------
_IS_TTY = sys.stdout.isatty()


def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def _green(s: str) -> str:
    return _ansi("32", s)


def _red(s: str) -> str:
    return _ansi("31", s)


def _yellow(s: str) -> str:
    return _ansi("33", s)


def _blue(s: str) -> str:
    return _ansi("34", s)


# ---------------------------------------------------------------------------
# Version comparison that does not pull in packaging.
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[.\-]?(rc|a|b|post)?(\d+))?")


def _parse_version(v: str) -> tuple[int, int, int, int, int]:
    """Parse version strings like ``0.5.6``, ``0.5.6rc1``, ``0.5.6.post1``."""
    m = _VERSION_RE.match(v.strip().lstrip("v"))
    if not m:
        return (0, 0, 0, 0, 0)
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    kind = m.group(4) or ""
    # Ordering: release (final) > post > rc > b > a.  We only care that
    # final >= rc for gating purposes, so encode final as +1 and pre-
    # release kinds as 0.
    rank = {"": 1, "post": 1, "rc": 0, "b": 0, "a": 0}.get(kind, 1)
    serial = int(m.group(5) or 0)
    return (major, minor, patch, rank, serial)


def _version_ge(a: str, b: str) -> bool:
    return _parse_version(a) >= _parse_version(b)


# ---------------------------------------------------------------------------
# Result accumulator.
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Helpers to read matrices without pandas.
# ---------------------------------------------------------------------------
def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            return next(reader)
        except StopIteration:
            return []


def _count_rows(path: Path) -> int:
    """Count data rows (excluding the header line) in a TSV."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _iter_column(path: Path, column: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if column not in (reader.fieldnames or []):
            return
        for row in reader:
            yield (row.get(column) or "").strip()


def _distinct_accessions(pr_matrix: Path) -> int:
    """Return number of distinct entries in the `Protein.Group` column."""
    seen: set[str] = set()
    for value in _iter_column(pr_matrix, "Protein.Group"):
        if not value:
            continue
        # A protein group can be a semicolon-separated list; count each
        # participating accession once.
        for piece in value.split(";"):
            piece = piece.strip()
            if piece:
                seen.add(piece)
    return len(seen)


# ---------------------------------------------------------------------------
# Core verifier.
# ---------------------------------------------------------------------------
def verify(
    output_dir: Path,
    *,
    min_version: str = "0.5.5",
    min_genes_frac: float = 0.9,
    min_pg_ratio: float = 0.5,
) -> VerifyResult:
    r = VerifyResult()

    if not output_dir.exists():
        r.fail(f"output_dir does not exist: {output_dir}")
        return r
    if not output_dir.is_dir():
        r.fail(f"output_dir is not a directory: {output_dir}")
        return r

    # -- 1. run_manifest.json --------------------------------------------------
    manifest_path = output_dir / "run_manifest.json"
    manifest: Optional[dict] = None
    if not manifest_path.exists():
        r.fail("run_manifest.json is missing (output was produced by "
               "diaquant < 0.5.4 or the manifest writer crashed silently)")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            r.note(f"run_manifest.json OK ({manifest_path.stat().st_size} bytes)")
        except json.JSONDecodeError as exc:
            r.fail(f"run_manifest.json is not valid JSON: {exc}")

    # -- 2. diaquant_version >= min_version -----------------------------------
    manifest_version = ""
    if manifest is not None:
        manifest_version = str(manifest.get("diaquant_version", "")).strip()
        if not manifest_version:
            r.fail("run_manifest.json has no `diaquant_version` field")
        elif not _version_ge(manifest_version, min_version):
            r.fail(
                f"diaquant_version={manifest_version!r} in manifest is older "
                f"than required {min_version!r}; your ptmquant image is "
                f"likely stale — re-pull ghcr.io/xformyx/ptmquant:latest"
            )
        else:
            r.note(f"diaquant_version={manifest_version} (>= {min_version})")

    # -- 3 + 4. pr_matrix / pg_matrix ----------------------------------------
    pr_matrix = output_dir / "report.pr_matrix.tsv"
    pg_matrix = output_dir / "report.pg_matrix.tsv"

    pr_rows = pg_rows = 0
    if not pr_matrix.exists():
        r.fail(f"{pr_matrix.name} is missing")
    else:
        pr_rows = _count_rows(pr_matrix)
        if pr_rows == 0:
            r.fail(f"{pr_matrix.name} is empty (0 data rows)")
        else:
            r.note(f"{pr_matrix.name}: {pr_rows} precursor rows")

    if not pg_matrix.exists():
        r.fail(f"{pg_matrix.name} is missing")
    else:
        pg_rows = _count_rows(pg_matrix)
        if pg_rows == 0:
            r.fail(f"{pg_matrix.name} is empty (0 data rows)")
        else:
            r.note(f"{pg_matrix.name}: {pg_rows} protein-group rows")

    # -- 5. pg_matrix Genes column populated ---------------------------------
    if pg_matrix.exists() and pg_rows > 0:
        populated = 0
        total = 0
        for value in _iter_column(pg_matrix, "Genes"):
            total += 1
            if value:
                populated += 1
        if total == 0:
            r.fail(f"{pg_matrix.name} has no `Genes` column (upgrade diaquant)")
        else:
            frac = populated / total
            if frac < min_genes_frac:
                r.fail(
                    f"{pg_matrix.name} `Genes` column only populated in "
                    f"{populated}/{total} rows ({frac:.1%}); required "
                    f">= {min_genes_frac:.0%}"
                )
            else:
                r.note(f"{pg_matrix.name} Genes populated: {frac:.1%}")

    # -- 6. pg_matrix rows >= min_pg_ratio * distinct pr accessions ---------
    if pr_matrix.exists() and pg_matrix.exists() and pr_rows > 0 and pg_rows > 0:
        distinct_pr = _distinct_accessions(pr_matrix)
        if distinct_pr > 0:
            ratio = pg_rows / distinct_pr
            if ratio < min_pg_ratio:
                r.fail(
                    f"pg_matrix has only {pg_rows} rows for {distinct_pr} "
                    f"distinct pr_matrix accessions (ratio={ratio:.2f} < "
                    f"{min_pg_ratio:.2f}); razor grouping / pg_quant filters "
                    f"may be dropping too many proteins"
                )
            else:
                r.note(
                    f"pg_matrix covers {ratio:.0%} of pr_matrix accessions "
                    f"({pg_rows}/{distinct_pr})"
                )

    # -- Advisory: predicted library / rescore / MBR -------------------------
    if manifest is not None:
        predicted = manifest.get("predicted_library", {}) or {}
        if predicted.get("applied") is False:
            r.warn(
                "run_manifest says predicted_library.applied=False; check "
                "peptdeep_status.txt in the image and `WITH_ALPHAPEPTDEEP` "
                "build arg"
            )
        elif predicted.get("applied") is True:
            libs = predicted.get("paths") or []
            r.note(
                f"predicted_library applied ({len(libs)} library file"
                f"{'s' if len(libs) != 1 else ''})"
            )

        rescore = manifest.get("rescoring", {}) or {}
        if rescore.get("applied") is False and rescore.get("configured") is True:
            r.warn("rescoring was configured but not applied at runtime")

        mbr = manifest.get("mbr", {}) or {}
        if mbr.get("applied") is True:
            n_res = int(mbr.get("n_rescued", 0))
            if n_res == 0:
                r.warn("MBR applied but rescued 0 PSMs — check RT tolerance "
                       "/ donor quorum settings")
            else:
                r.note(f"MBR rescued {n_res} PSMs")
        elif mbr.get("configured") is True and mbr.get("applied") is False:
            r.warn("MBR was configured but did not run")

    # -- Advisory: ptm_site_matrix -------------------------------------------
    site_matrix = output_dir / "report.ptm_site_matrix.tsv"
    if site_matrix.exists():
        n_sites = _count_rows(site_matrix)
        header = _read_header(site_matrix)
        if n_sites == 0:
            # Only warn if a PTM-aware pass was configured (we cannot know
            # without the config, but manifest often records this).
            ptm_cfg = bool(manifest and manifest.get("passes"))
            msg = (
                f"{site_matrix.name} is empty (0 data rows)"
                + (
                    " even though PTM passes were configured; this is the "
                    "v0.5.3/v0.5.4 regression and means the running image "
                    "is still pre-v0.5.5"
                    if ptm_cfg
                    else ""
                )
            )
            r.warn(msg)
        else:
            r.note(
                f"{site_matrix.name}: {n_sites} sites, "
                f"columns={len(header)}"
            )

    return r


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _default_min_version() -> str:
    """Pick up the running diaquant's own version if importable, else 0.5.5."""
    try:
        import diaquant  # type: ignore

        return getattr(diaquant, "__version__", "0.5.5")
    except Exception:
        return "0.5.5"


def _format_report(r: VerifyResult) -> str:
    lines: list[str] = []
    if r.info:
        lines.append(_blue("INFO:"))
        lines.extend(f"  - {m}" for m in r.info)
    if r.warnings:
        lines.append(_yellow("WARN:"))
        lines.extend(f"  - {m}" for m in r.warnings)
    if r.failures:
        lines.append(_red("FAIL:"))
        lines.extend(f"  - {m}" for m in r.failures)
    if r.ok:
        lines.append(_green("PASS — PTMQuant output verified"))
    else:
        lines.append(_red(f"FAIL — {len(r.failures)} mandatory check(s) failed"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a PTMQuant output directory against v0.5.5+ expectations."
    )
    parser.add_argument("output_dir", type=Path, help="Path to PTMQuant output directory")
    parser.add_argument(
        "--min-version",
        default=_default_min_version(),
        help="Minimum required diaquant version (default: running package version or 0.5.5)",
    )
    parser.add_argument(
        "--min-genes-frac",
        type=float,
        default=0.9,
        help="Minimum fraction of pg_matrix rows that must have Genes populated (default: 0.9)",
    )
    parser.add_argument(
        "--min-pg-ratio",
        type=float,
        default=0.5,
        help="pg_matrix rows must be at least this fraction of pr_matrix distinct accessions (default: 0.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report on stdout instead of the coloured human summary",
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    if not out_dir.exists():
        print(f"Output dir does not exist: {out_dir}", file=sys.stderr)
        return 2

    result = verify(
        out_dir,
        min_version=args.min_version,
        min_genes_frac=args.min_genes_frac,
        min_pg_ratio=args.min_pg_ratio,
    )

    if args.json:
        print(json.dumps(
            {
                "ok": result.ok,
                "failures": result.failures,
                "warnings": result.warnings,
                "info": result.info,
            },
            indent=2,
        ))
    else:
        print(_format_report(result))

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
