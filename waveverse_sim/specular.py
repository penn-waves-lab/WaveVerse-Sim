"""Specular returns from a fresh Sionna geometry and field solve."""

import numpy as np
import tensorflow as tf

from . import cuda_mixer


def specular_signals(scene, spec, num_samples, max_depth, rng=None, moving_object=None):
    """Return coherent/control signals, (1, RX*TX, ADC), and valid path count.

    Devices and meshes must already be positioned for this chirp. No path or
    signal is retained across calls. A control randomizes all specular paths
    for an aperture, or only paths touching moving_object for Doppler.
    """
    traced = scene.trace_paths(
        max_depth=max_depth,
        num_samples=num_samples,
        los=False,
        reflection=True,
        scattering=False,
        diffraction=False,
        ris=False,
    )
    count = int(np.asarray(traced[0].mask).sum())
    paths = scene.compute_fields(*traced, scat_random_phases=False, check_scene=False)
    paths.normalize_delays = False
    alpha, tau = paths.cir(
        los=False, reflection=True, scattering=False, diffraction=False, ris=False
    )
    nrx, ntx, npaths = int(alpha.shape[2]), int(alpha.shape[4]), int(alpha.shape[5])
    if not npaths:
        zero = np.zeros((1, nrx * ntx, spec.adc_samples), np.complex64)
        return zero, zero.copy() if rng is not None else None, count
    a = tf.reshape(alpha, [1, nrx * ntx, npaths])
    delays = tf.reshape(tau, [1, npaths])
    coherent = np.asarray(cuda_mixer.mix(a, delays, spec, split=1))
    control = None
    if rng is not None:
        moving = np.ones(npaths, bool)
        if moving_object is not None:
            objects = np.asarray(paths.objects)[:, 0, 0]
            moving = (objects == scene.get(moving_object).object_id).any(0)
        phases = np.ones(npaths, np.complex64)
        phases[moving] = np.exp(1j * rng.uniform(-np.pi, np.pi, int(moving.sum())))
        control = np.asarray(
            cuda_mixer.mix(a * tf.constant(phases[None, None]), delays, spec, split=1)
        )
    return coherent, control, count
