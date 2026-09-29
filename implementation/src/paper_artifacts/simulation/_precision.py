"""Select the arithmetic precision of the multislice propagation.

PySlice defaults to ``float64``/``complex128`` on every device except MPS. That
default is expensive on consumer and workstation Ampere, where the
double-precision rate is a small fraction of single. On the RTX A5000 used here
a batched ``fft2`` is 5.3x slower in complex128 than in complex64, and the whole
simulation runs 3.8x slower end to end, 46.3 ms per probe position against 12.2.

Nothing is bought with it. Measured on this specimen, single against double: the
ADF fraction differs by 1.5e-7 relative, lambda by 9e-6 of its peak, and the flux
ratio by 1.2e-6. All sit orders of magnitude below the 3% spread across
frozen-phonon seeds that is already present in the simulation. Single precision
is what abTEM and Prismatic use by default, for the same reason.

The choice is made through PySlice's own ``PYSLICE_PRECISION`` environment
variable rather than by patching, because PySlice constructs its backends
internally with no arguments, so the environment is the only seam that reaches
them.
"""

from __future__ import annotations

import os

_DTYPES: dict[str, str] = {"single": "complex64", "double": "complex128"}


def apply(kind: str = "single") -> None:
    """
    Ask PySlice for the requested precision.

    Parameters
    ----------
    kind : {"single", "double"}, optional
        ``"single"`` gives float32/complex64, ``"double"`` float64/complex128.
        Default ``"single"``.

    Raises
    ------
    ValueError
        If ``kind`` is not recognised.

    Notes
    -----
    Call before constructing any backend; :mod:`paper_artifacts.simulation` does
    so at import. MPS is left alone by PySlice itself, since it cannot do
    float64 at all.
    """
    if kind not in _DTYPES:
        raise ValueError(f"precision must be one of {sorted(_DTYPES)}, got {kind!r}")
    os.environ["PYSLICE_PRECISION"] = kind


def current() -> str:
    """
    Report the precision a fresh backend would use.

    Returns
    -------
    str
        ``"single"``, ``"double"``, or ``"unknown"`` if PySlice is absent.
    """
    try:
        from pyslice.backend import make_backend
    except ImportError:
        return "unknown"
    complex_dtype = str(make_backend().complex_dtype)
    for kind, name in _DTYPES.items():
        if name in complex_dtype:
            return kind
    return "unknown"
