"""Orbitrap instrument presets for diaquant (NEW in 0.5.1).

Different Orbitrap generations have meaningfully different MS1/MS2 mass
accuracy, usable precursor m/z range and HCD collision energy.  Requiring the
end user to tune seven separate YAML knobs (``precursor_tol_ppm``,
``fragment_tol_ppm``, ``library_precursor_tol_ppm``,
``library_fragment_tol_ppm``, ``min_precursor_mz``, ``max_precursor_mz``,
``pred_lib_nce``, ``pred_lib_instrument``) for every new instrument is both
tedious and error-prone, so 0.5.1 ships a small catalog of curated
:class:`InstrumentPreset` objects.

Design principles:

1. **Presets are optional.**  When ``DiaQuantConfig.instrument`` is left at
   the default (``"exploris_240"``) the numeric values match the 0.4.x
   defaults exactly, so every 0.5.0 YAML keeps producing identical results.
2. **YAML-level values win.**  ``apply_preset`` only overwrites *default*
   (0.4.x) values.  If the user has explicitly set, say,
   ``precursor_tol_ppm: 4.0`` in the YAML, the preset leaves it alone.  This
   avoids the common frustration of "why is my carefully tuned tolerance
   being ignored" when adopting a new feature.
3. **Small on purpose.**  Per the user's request we ship only four presets
   (Exploris 240, Orbitrap Astral, Orbitrap Eclipse, Fusion Lumos) rather
   than the full ten-instrument menu.  Adding new entries is a one-line
   change.

Every preset also records the AlphaPeptDeep instrument tag (``peptdeep``):
AlphaPeptDeep was trained on labelled RT/MS2 data from specific instrument
classes, and passing the closest matching tag (e.g. ``Lumos`` for Eclipse /
Astral) yields noticeably better fragment-intensity predictions than the
generic default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class InstrumentPreset:
    """Numeric defaults + AlphaPeptDeep tag for one Orbitrap generation."""

    name: str                                # canonical id used in YAML / CLI
    display_name: str                        # pretty label shown by list-instruments
    precursor_tol_ppm: float                 # MS1 identification tolerance
    fragment_tol_ppm: float                  # MS2 identification tolerance
    library_precursor_tol_ppm: float         # MS1 tolerance for library generation (slightly looser)
    library_fragment_tol_ppm: float          # MS2 tolerance for library generation
    min_precursor_mz: float
    max_precursor_mz: float
    pred_lib_nce: float                      # normalised HCD collision energy
    pred_lib_instrument: str                 # AlphaPeptDeep instrument tag
    description: str


# The baseline that reproduces diaquant 0.4.x / 0.5.0 defaults verbatim.  This
# is important: because ``apply_preset`` only overwrites *default* fields,
# every numeric value here must match DiaQuantConfig's 0.4.x default or we
# would silently change the behaviour of existing configurations.
_EXPLORIS_240 = InstrumentPreset(
    name="exploris_240",
    display_name="Thermo Orbitrap Exploris 240",
    precursor_tol_ppm=6.0,
    fragment_tol_ppm=12.0,
    library_precursor_tol_ppm=8.0,
    library_fragment_tol_ppm=15.0,
    min_precursor_mz=400.0,
    max_precursor_mz=1000.0,
    pred_lib_nce=28.0,
    pred_lib_instrument="QE",
    description=(
        "Exploris 240, narrow DIA (6 m/z). Default preset; values match diaquant 0.4.x / 0.5.0 defaults."
    ),
)


INSTRUMENT_PRESETS: Dict[str, InstrumentPreset] = {
    _EXPLORIS_240.name: _EXPLORIS_240,
    "orbitrap_astral": InstrumentPreset(
        name="orbitrap_astral",
        display_name="Thermo Orbitrap Astral",
        precursor_tol_ppm=3.0,
        fragment_tol_ppm=8.0,
        library_precursor_tol_ppm=5.0,
        library_fragment_tol_ppm=10.0,
        min_precursor_mz=380.0,
        max_precursor_mz=980.0,
        pred_lib_nce=27.0,
        pred_lib_instrument="Lumos",
        description=(
            "Astral analyser, highest mass accuracy; tighter MS1/MS2 tolerances exploit the 2 ppm MS1 spec."
        ),
    ),
    "orbitrap_eclipse": InstrumentPreset(
        name="orbitrap_eclipse",
        display_name="Thermo Orbitrap Eclipse Tribrid",
        precursor_tol_ppm=5.0,
        fragment_tol_ppm=10.0,
        library_precursor_tol_ppm=7.0,
        library_fragment_tol_ppm=12.0,
        min_precursor_mz=350.0,
        max_precursor_mz=1500.0,
        pred_lib_nce=30.0,
        pred_lib_instrument="Lumos",
        description=(
            "Eclipse Tribrid; wider precursor range for top-down / middle-down work, NCE tuned for TurboTMT compatibility."
        ),
    ),
    "fusion_lumos": InstrumentPreset(
        name="fusion_lumos",
        display_name="Thermo Orbitrap Fusion Lumos",
        precursor_tol_ppm=5.0,
        fragment_tol_ppm=12.0,
        library_precursor_tol_ppm=8.0,
        library_fragment_tol_ppm=15.0,
        min_precursor_mz=350.0,
        max_precursor_mz=1500.0,
        pred_lib_nce=30.0,
        pred_lib_instrument="Lumos",
        description=(
            "Fusion Lumos; legacy Orbitrap baseline. Slightly looser than Eclipse to account for older calibrations."
        ),
    ),
}


# --- helper: fields that ``apply_preset`` may overwrite -------------------
# We key off the 0.4.x default value.  If the current DiaQuantConfig value
# equals the default, we treat it as "user did not set this" and let the
# preset take effect.  Any other value means the user set it explicitly and
# must be respected.
_PRESET_FIELDS = (
    "precursor_tol_ppm",
    "fragment_tol_ppm",
    "library_precursor_tol_ppm",
    "library_fragment_tol_ppm",
    "min_precursor_mz",
    "max_precursor_mz",
    "pred_lib_nce",
    "pred_lib_instrument",
)


def get_instrument(name: str) -> InstrumentPreset:
    """Return the :class:`InstrumentPreset` for ``name`` or raise with a list."""
    key = name.strip().lower()
    if key not in INSTRUMENT_PRESETS:
        known = ", ".join(sorted(INSTRUMENT_PRESETS))
        raise ValueError(
            f"Unknown instrument '{name}'. Valid choices: {known}."
        )
    return INSTRUMENT_PRESETS[key]


def list_instruments() -> Dict[str, InstrumentPreset]:
    """Return the entire catalog (used by ``diaquant list-instruments``)."""
    return dict(INSTRUMENT_PRESETS)


def apply_preset(cfg, preset: InstrumentPreset) -> None:  # type: ignore[no-untyped-def]
    """Overwrite *only* the default-valued fields of ``cfg`` with ``preset``.

    ``cfg`` is typed as ``Any`` to avoid an import cycle with ``config.py``;
    in practice it is always a :class:`diaquant.config.DiaQuantConfig`.
    """
    # Pull the 0.4.x defaults from the exploris_240 baseline (which by
    # construction matches DiaQuantConfig's field defaults).  We only
    # overwrite when the current value equals the default.
    baseline = _EXPLORIS_240
    for field_name in _PRESET_FIELDS:
        current = getattr(cfg, field_name, None)
        default = getattr(baseline, field_name)
        if current is None or current == default:
            setattr(cfg, field_name, getattr(preset, field_name))
