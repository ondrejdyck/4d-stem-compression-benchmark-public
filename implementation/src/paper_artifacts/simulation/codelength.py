"""Code length of Poisson-sampled counts, given the rate that produced them.

The manuscript contrasts three things: what a general-purpose lossless codec
achieves on a simulated 4D-STEM datacube, what the shortest possible code
achieves given an exact model of the *instrument*, and how few bytes were used
to generate the cube in the first place.

This module supplies the middle term. Given the noiseless expected intensity
``lam`` at every detector pixel, the counts are independent Poisson draws, so
the shortest code for them has expected length

    sum_i H(Poisson(lam_i))

bits, and the length actually achieved on one realisation is

    sum_i -log2 P(counts_i | lam_i).

Both are computed here. They differ by a sampling fluctuation of order
sqrt(N), so agreeing to a few parts in 10^4 over 4x10^8 pixels is the
expected behaviour and is checked in the driver.

Neither quantity is reachable in an experiment, because ``lam`` is what one
would have if the answer were already known. It is reported as a bound: the
best any codec could do knowing the detector physics exactly and nothing at
all about the specimen.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

LOG2E = float(np.log2(np.e))


def _kmax(lam_max: float) -> int:
    """Truncation for the Poisson pmf sum: mean plus eight standard deviations."""
    return int(max(12, np.ceil(lam_max + 8.0 * np.sqrt(lam_max) + 8.0)))


def poisson_entropy_bits(lam: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Entropy of Poisson(``lam``) in bits, elementwise.

    Summed by the recurrence ``p_k = p_{k-1} * lam / k`` rather than by calling
    a pmf, which is what makes this affordable at 4x10^8 pixels.
    """
    lam = np.asarray(lam, dtype=np.float64)
    if kmax is None:
        kmax = _kmax(float(lam.max()) if lam.size else 0.0)
    p = np.exp(-lam)
    h = np.zeros_like(lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        h -= np.where(p > 0, p * np.log2(p), 0.0)
        for k in range(1, kmax + 1):
            p = p * lam / k
            h -= np.where(p > 0, p * np.log2(p), 0.0)
    return h


def poisson_neglogp_bits(lam: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``-log2 P(counts | lam)`` in bits, elementwise.

    This is the length an ideal arithmetic coder holding ``lam`` would actually
    emit for this realisation, as opposed to the expected length.
    """
    lam = np.asarray(lam, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    out = np.empty_like(lam)
    zero = lam <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (lam - c * np.log(np.where(zero, 1.0, lam)) + gammaln(c + 1.0)) * LOG2E
    # lam == 0 codes c == 0 for free and c > 0 not at all.
    out = np.where(zero, np.where(c > 0, np.inf, 0.0), out)
    return out
