"""WaveVerse RF simulation; GPU dependencies are loaded only when needed."""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "FMCWSimulator": "fmcw",
    "RadarSpec": "radar_spec",
    "PANORADAR_SPEC": "radar_spec",
    "SpatialAperture": "spatial_aperture",
    "TemporalBurst": "temporal_burst",
    "HumanVertexGroups": "grouped_scattering",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
