"""NumPy-only FMCW range/Doppler processing; no GPU initialization on import."""

import numpy as np


def range_doppler(signal, mti=False, weighted_mti=False):
    """(chirps, channels, ADC) -> (Doppler bins, range bins) power.

    Raw processing is the default. Both optional mean filters suppress real
    stationary returns as well as clutter; only weighted MTI forces the Hann
    window's zero-Doppler bin to zero.
    """
    s = np.asarray(signal, np.complex128)
    if s.ndim != 3 or min(s.shape) < 1 or not np.isfinite(s).all():
        raise ValueError("Expected a nonempty (chirps, channels, ADC) signal")
    if mti and not weighted_mti:
        s = s - s.mean(axis=0, keepdims=True)
    fast_window = np.hanning(s.shape[-1]) if s.shape[-1] > 2 else np.ones(s.shape[-1])
    rp = np.fft.fft(s * fast_window, axis=-1)
    window = (np.hanning(len(s)) if len(s) > 2 else np.ones(len(s)))[:, None, None]
    if mti and weighted_mti:
        rp -= (rp * window).sum(axis=0, keepdims=True) / window.sum()
    doppler = np.fft.fftshift(np.fft.fft(rp * window, axis=0), axes=0)
    return np.sum(np.abs(doppler) ** 2, axis=1)


def range_mask(ranges, gate):
    """Validate a range selection and return its sampled-bin mask."""
    gate = np.asarray(gate, dtype=float)
    if gate.shape != (2,) or not np.isfinite(gate).all() or not 0 <= gate[0] < gate[1]:
        raise ValueError("Range gate must be finite, nonnegative and increasing")
    ranges = np.asarray(ranges)
    if ranges.ndim != 1 or not len(ranges) or not np.isfinite(ranges).all():
        raise ValueError("Expected a nonempty, finite range axis")
    roi = (ranges >= gate[0]) & (ranges <= gate[1])
    if not roi.any():
        raise ValueError("Range gate contains no sampled range bins")
    return roi


def range_spectra(rd, ranges, gate):
    """Integrate selected range bins without altering the zero-Doppler bin."""
    rd = np.asarray(rd)
    roi = range_mask(ranges, gate)
    if rd.ndim != 3 or min(rd.shape) < 1 or rd.shape[-1] != len(roi):
        raise ValueError(
            "Expected (frames, Doppler bins, range bins) matching the range axis"
        )
    if not np.isfinite(rd).all() or (rd < 0).any():
        raise ValueError("Range-Doppler power must be finite and nonnegative")
    return rd[:, :, roi].sum(axis=2).T


def velocity_limits(velocities, truth, limit=None):
    """Choose a symmetric view that contains the prescribed motion when unaliased."""
    if limit is not None and (not np.isfinite(limit) or limit <= 0):
        raise ValueError("velocity-limit must be finite and positive")
    if limit is None:
        limit = min(
            max(3.0, 1.2 * float(np.max(np.abs(truth)))),
            float(np.max(np.abs(velocities))),
        )
    return [-float(limit), float(limit)]
