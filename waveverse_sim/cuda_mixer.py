"""CUDA FMCW mixing with shared-memory path tiles and register accumulation.

Sample-independent phase is reduced modulo one cycle in float64 before
float32 accumulation. CuPy and NVRTC are required; the aperture uses the
TensorFlow mixer when CUDA kernel compilation is unavailable.
"""

import numpy as np
import tensorflow as tf
from .cuda_runtime import prepare_nvrtc, load_kernel

MAX_CHANNELS = 8  # accumulators held in registers per thread

_kernel = None
_prepare_kernel = None
_error = None


def available():
    global _kernel, _error
    if _kernel is not None:
        return True
    if _error is not None:
        return False
    try:
        prepare_nvrtc()
        import cupy as cp

        kernel = cp.RawKernel(load_kernel("fmcw.cu"), "mix_kernel")
        kernel.compile()
        _kernel = kernel
    except Exception as e:  # missing cupy, no NVRTC, no GPU
        _error = e
        return False
    return True


def prepare_weights(alpha, tau, spec):
    """Fuse the sample-independent phase calculation and channel weighting.

    Conjugated CIR gains already supply the positive carrier-delay phase.
    Float64 phase reduction limits numerical error before float32 accumulation.
    """
    global _prepare_kernel
    import cupy as cp
    from tensorflow.experimental import dlpack as tfdl

    if _prepare_kernel is None:
        _prepare_kernel = cp.RawKernel(
            load_kernel("fmcw.cu"), "prepare_weights", options=("--fmad=false",)
        )
    # TensorFlow's capsule export makes the producing stream ready for CuPy.
    ac = cp.from_dlpack(tfdl.to_dlpack(alpha))
    tc = cp.from_dlpack(tfdl.to_dlpack(tau))
    K, C, P = map(int, alpha.shape)
    W = cp.empty((K, C, P), cp.complex64)
    x = cp.empty((K, P), cp.float32)
    _prepare_kernel(
        ((K * P + 255) // 256,),
        (256,),
        (
            ac,
            tc,
            W,
            x,
            np.int32(P),
            np.int32(C),
            np.int64(K * P),
            np.float64(spec.slope * 1e12),
            np.float64(spec.adc_start_time * 1e-6),
            np.float64(1.0 / (spec.sample_rate * 1e3)),
        ),
    )
    return x, W


def mix(alpha, tau, spec, split=16, threads=256):
    """Beat signals for a batch of aperture positions.

    Args:
        alpha: (K, C, P) complex64 tf tensor on the GPU, C = n_rx * n_tx
            channels in (rx, tx) row-major order, Sionna's cir() convention
            (propagation phase included).
        tau: (K, P) float32 delays.
    Returns: (K, C, n_samples) complex64 tf tensor.
    """
    if not available():
        raise RuntimeError("CUDA mixer unavailable") from _error
    import cupy as cp
    from tensorflow.experimental import dlpack as tfdl

    K, C, P = (int(v) for v in alpha.shape)
    N = spec.adc_samples
    if not 1 <= C <= MAX_CHANNELS or split < 1 or not 1 <= N <= threads <= 1024:
        raise ValueError(
            "CUDA mixer requires 1–8 channels, positive splits, and ADC samples <= threads <= 1024"
        )
    xc, Wc = prepare_weights(alpha, tau, spec)
    S = cp.empty((K, split, C, N), dtype=cp.complex64)
    _kernel(
        (K, split),
        (threads,),
        (xc, Wc, S, np.int32(P), np.int32(C), np.int32(N), np.int32(split)),
        shared_mem=threads * 4 + C * threads * 8,
    )
    cp.cuda.runtime.deviceSynchronize()
    try:  # TensorFlow accepts unversioned DLPack capsules
        St = tfdl.from_dlpack(S.__dlpack__())
    except Exception:
        St = tf.constant(cp.asnumpy(S))
    return tf.reduce_sum(St, axis=1)
