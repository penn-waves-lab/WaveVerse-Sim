"""Locate pip-installed NVRTC without a machine-specific LD_LIBRARY_PATH."""

import ctypes
from functools import lru_cache
from importlib import metadata, util
from pathlib import Path

_handles = []
_ready = False


def prepare_nvrtc():
    """Preload the matching NVIDIA wheel's compiler, if installed.

    CuPy 13 does not always discover nvidia-cuda-nvrtc-cu12 on Linux. Loading
    its SONAME once by absolute path makes it available to CuPy's dlopen call.
    Existing CUDA toolkit installations remain supported when no wheel exists.
    """
    global _ready
    if _ready:
        return
    installed = []
    for major in (11, 12):
        try:
            metadata.version(f"cupy-cuda{major}x")
            installed.append(major)
        except metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        raise RuntimeError("Install only one CuPy CUDA wheel in this environment")
    major = installed[0] if installed else None
    try:
        package = util.find_spec("nvidia.cuda_nvrtc")
    except ModuleNotFoundError:
        package = None
    if package is not None and major is not None:
        for location in package.submodule_search_locations or []:
            libraries = Path(location) / "lib"
            nvrtc = sorted(libraries.glob(f"libnvrtc.so.{major}*"))
            if nvrtc:
                for builtin in sorted(libraries.glob(f"libnvrtc-builtins.so.{major}*")):
                    _handles.append(ctypes.CDLL(str(builtin), mode=ctypes.RTLD_GLOBAL))
                _handles.append(ctypes.CDLL(str(nvrtc[0]), mode=ctypes.RTLD_GLOBAL))
                break
    _ready = True


@lru_cache(maxsize=None)
def load_kernel(name):
    """Read a bundled CUDA kernel for runtime compilation."""
    return (Path(__file__).parent / "kernels" / name).read_text(encoding="utf-8")
