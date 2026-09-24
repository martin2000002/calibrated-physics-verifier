"""Thread limits for numeric libraries and Blender (runtime.threads)."""

from __future__ import annotations

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def thread_env(threads: int) -> dict[str, str]:
    """Environment variables that cap native thread pools; pass them to child processes."""
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return {name: str(threads) for name in THREAD_VARIABLES}
