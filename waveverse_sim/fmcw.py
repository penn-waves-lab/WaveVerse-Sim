from typing import Literal

import numpy as np
import tensorflow as tf

from .radar_spec import RadarSpec


class FMCWSimulator:
    """
    Memory-optimized FMCW simulator using XLA compilation and batched processing.
    """

    def __init__(
        self,
        radar_spec: RadarSpec,
        dtype: Literal["float32", "float64"] = "float32",
    ):
        self.spec = radar_spec
        self.dtype = dtype
        self._t_sample = None

    @property
    def tf_dtype(self):
        return tf.float32 if self.dtype == "float32" else tf.float64

    @property
    def tf_complex_dtype(self):
        return tf.complex64 if self.dtype == "float32" else tf.complex128

    def _initialize_time_samples(self):
        """Initialize time samples tensor once and cache it"""
        if self._t_sample is not None:
            return self._t_sample

        t_sample = (
            tf.range(0, self.spec.adc_samples, dtype=self.tf_dtype)
            / (self.spec.sample_rate * 1e3)
            + self.spec.adc_start_time * 1e-6
        )
        t_sample = tf.reshape(t_sample, (1,) * 7 + (-1,))
        t_sample = tf.tile(t_sample, [1] * 6 + [self.spec.chirp_per_frame] + [1])

        self._t_sample = t_sample
        return t_sample

    def _prepare_alpha_tau(self, alpha, tau):
        """Validate and cast inputs; return (alpha, tau, n_paths) without any global padding."""
        if alpha.ndim != 7:
            raise ValueError(
                f"Alpha should have 7 dimensions, but got {alpha.ndim} dimensions."
            )

        n_frames, n_rx, n_rx_ants, n_tx, n_tx_ants, n_paths, n_chirps = alpha.shape
        if n_chirps != self.spec.chirp_per_frame:
            raise ValueError(
                f"Alpha should have {self.spec.chirp_per_frame} chirps, but got {n_chirps} chirps."
            )

        # Delays may be shared across chirps or specified per chirp.
        if tau.ndim in (4, 5):
            tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        if tau.ndim == 6:
            tau = tf.expand_dims(tau, axis=-1)  # chirp axis

        alpha = tf.cast(alpha, self.tf_complex_dtype)
        tau = tf.cast(tau, self.tf_dtype)

        alpha = tf.expand_dims(alpha, axis=-1)
        tau = tf.expand_dims(tau, axis=-1)

        return alpha, tau, n_paths

    @tf.function(jit_compile=True)
    def _process_one_batch(self, alpha_batch, tau_batch, t_sample):
        """Process a single batch with fixed shape."""
        # alpha already contains exp(-j*2*pi*fc*tau). TX*conj(RX) adds
        # only the chirp-delay term; adding fc*tau here would count it twice.
        delay = tf.cast(tau_batch, tf.float64)
        sample = tf.cast(t_sample, tf.float64)
        phase = 2 * np.pi * self.spec.slope * 1e12 * delay * (sample - 0.5 * delay)
        phase = tf.math.floormod(phase, 2 * np.pi)
        unit = tf.cast(tf.complex(tf.cos(phase), tf.sin(phase)), self.tf_complex_dtype)
        return tf.reduce_sum(tf.math.conj(alpha_batch) * unit, axis=5)

    def simulate_propagation(self, alpha, tau, path_batch_size=100) -> tf.Tensor:
        """
        Memory-optimized simulation using a Python-level loop over path batches.

        Each batch is sliced directly from alpha/tau without pre-padding the full
        tensor, keeping peak GPU memory usage to one batch at a time.
        """
        if path_batch_size < 1:
            raise ValueError("path_batch_size must be positive")
        alpha, tau, n_paths = self._prepare_alpha_tau(alpha, tau)
        t_sample = self._initialize_time_samples()

        acc_shape = tf.concat(
            [
                tf.shape(alpha)[:5],
                [self.spec.chirp_per_frame],
                [self.spec.adc_samples],
            ],
            axis=0,
        )
        mixed_signal_sum = tf.zeros(acc_shape, dtype=self.tf_complex_dtype)

        for start in range(0, n_paths, path_batch_size):
            end = min(start + path_batch_size, n_paths)
            a_slice = alpha[:, :, :, :, :, start:end, :, :]
            t_slice = tau[:, :, :, :, :, start:end, :, :]

            # Pad the last partial batch to a fixed size for JIT compatibility
            chunk = end - start
            if chunk < path_batch_size:
                pad_amt = path_batch_size - chunk
                pad_spec = [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, pad_amt],
                    [0, 0],
                    [0, 0],
                ]
                a_slice = tf.pad(a_slice, pad_spec)
                t_slice = tf.pad(t_slice, pad_spec)

            mixed_signal_sum += self._process_one_batch(a_slice, t_slice, t_sample)

        return mixed_signal_sum
