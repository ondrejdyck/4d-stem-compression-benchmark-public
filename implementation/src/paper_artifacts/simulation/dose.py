"""Dose and Poisson counting statistics.

This is where the simulator stops being deterministic physics and becomes
data. Everything upstream is an exact intensity; everything downstream sees
integer counts and must infer the intensity back out.

**PySlice's own sampler is not the one needed here.**
``WFData.counts(N)`` normalises intensity to a probability over the entire
``(probe, frame, kx, ky, layer)`` array and draws exactly ``N`` uniform
randoms, which is a multinomial. The total is then fixed rather than
Poisson, pixels are weakly anti-correlated rather than independent, and --
worst for a scan -- normalising across the probe axis couples probe
positions to one another. The manuscript's argument rests on counts being
independent given lambda with Fano factor 1, so the Poisson path lives here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from paper_artifacts.simulation import config
from paper_artifacts.simulation.diffraction import Diffraction


def electrons_per_probe(
    current_A: float = config.BEAM_CURRENT_A,
    dwell_s: float = config.DWELL_S,
) -> float:
    # Why this exists: the dose is the single number that sets how hard the
    # inference problem is, and it is easy to state loosely ("low dose") and
    # then compare against a measurement taken at a different one. Naming it
    # keeps every signal-to-noise claim in the project anchored to a current
    # and a dwell rather than to a habit.
    """
    Mean number of electrons incident during one probe dwell.

    Parameters
    ----------
    current_A : float, optional
        Beam current in amperes. Default :data:`paper_artifacts.simulation.config.BEAM_CURRENT_A`.
    dwell_s : float, optional
        Dwell time in seconds. Default :data:`paper_artifacts.simulation.config.DWELL_S`.

    Returns
    -------
    float
        Expected electron count. Not an integer: it is the mean of a Poisson
        process, and rounding it here would quietly remove the dose's own
        shot noise.

    Examples
    --------
    >>> round(electrons_per_probe(60e-12, 1e-6), 1)
    374.5
    """
    return current_A * dwell_s / config.ELEMENTARY_CHARGE_C


def collected_fraction(diff: Diffraction, position: int = 0) -> float:
    # Why this exists: as a sanity check on the grid, and to make the size of
    # the band-limit loss visible rather than implicit. It is NOT a physical
    # observable and nothing downstream should treat it as one -- see Notes.
    """
    Fraction of the incident beam still represented on the detector grid.

    Parameters
    ----------
    diff : Diffraction
        Patterns from :func:`paper_artifacts.simulation.diffraction.simulate`.
    position : int, optional
        Probe-position index. Default 0.

    Returns
    -------
    float
        Collected intensity over incident intensity. About 0.983 on a
        tungsten column, 0.9997 at a hollow site.

    Notes
    -----
    The shortfall is intensity scattered past the multislice anti-aliasing
    aperture at 2/3 of Nyquist -- 180.3 mrad on this grid -- which the
    propagator applies at every slice. The 180 mrad detector crop removes
    nothing further: measured at exactly zero, because the anti-aliasing
    aperture has already zeroed everything outside it.

    This is a band-limit loss, not absorption, and it is **not** what any
    virtual detector here reads. A bright-field detector's contrast comes
    from scattering out of its own 30 mrad disk into 30-180 mrad, which is
    about 18% of the beam at a tungsten column -- an order of magnitude
    larger than this, and a different quantity.
    """
    intensity = np.asarray(diff.intensity[position], dtype=np.float64)
    return float(intensity.sum() / diff.incident_total)


def expected_counts(
    diff: Diffraction,
    n_electrons: Optional[float] = None,
    position: int = 0,
) -> np.ndarray:
    # Why this exists: lambda is the quantity the Bayesian model is trying to
    # recover, so it needs one unambiguous definition. The subtle choice is
    # the denominator -- see Notes -- and making it here once means no later
    # stage can normalise the missing electrons back in without noticing.
    """
    Expected counts per detector pixel for one probe position.

    Parameters
    ----------
    diff : Diffraction
        Patterns from :func:`paper_artifacts.simulation.diffraction.simulate`.
    n_electrons : float, optional
        Dose. Default :func:`electrons_per_probe`.
    position : int, optional
        Probe-position index. Default 0.

    Returns
    -------
    numpy.ndarray
        ``(nkx, nky)`` array of Poisson rates, summing to the dose times
        :func:`collected_fraction`.

    Notes
    -----
    Scaled by the *incident* total, not by the pattern's own sum. Dividing
    by the pattern sum would assert that every electron lands somewhere on
    the detector, which is false: about 1.7% scatters past the multislice
    anti-aliasing aperture at a tungsten column. The detector crop removes
    nothing further on this grid, since the anti-aliasing aperture already
    has. The shortfall is kept rather than normalised away.

    ``sum(lambda)`` is therefore position-dependent. Note this is a
    band-limit loss rather than a physical one; see
    :func:`collected_fraction`.
    """
    dose = electrons_per_probe() if n_electrons is None else n_electrons
    # float64 regardless of the propagation precision: lambda is summed over
    # thousands of pixels downstream and is the rate the counts are drawn at.
    intensity = np.asarray(diff.intensity[position], dtype=np.float64)
    return intensity * (dose / diff.incident_total)


def make_rng(seed: int) -> np.random.Generator:
    # Why this exists: to make reaching for global NumPy state impossible
    # rather than merely discouraged. A run that depends on whatever drew
    # random numbers before it is irreproducible in a way no test catches,
    # because it never fails the same way twice.
    """
    Construct an explicitly seeded generator.

    Parameters
    ----------
    seed : int
        Seed. Required; there is no default.

    Returns
    -------
    numpy.random.Generator
        PCG64 generator.
    """
    return np.random.default_rng(seed)


def sample_counts(lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Why this exists: one line of NumPy, but it is the line that makes the
    # counts independent Poisson draws given lambda, which is what the
    # code-length comparison assumes. PySlice's own sampler does not.
    """
    Draw Poisson counts at the given rates.

    Parameters
    ----------
    lam : numpy.ndarray
        Expected counts per pixel, from :func:`expected_counts`.
    rng : numpy.random.Generator
        Generator from :func:`make_rng`.

    Returns
    -------
    numpy.ndarray
        Integer counts, same shape as ``lam``. Pixels are independent given
        ``lam`` and the total is itself Poisson, so the dose carries its own
        shot noise rather than being fixed.

    Examples
    --------
    >>> counts = sample_counts(np.full(4, 5.0), make_rng(0))
    >>> counts.shape, counts.dtype.kind
    ((4,), 'i')
    """
    return rng.poisson(lam)


def sample_counts_multinomial(
    lam: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    # Why this exists: only as the thing our sampler is compared against. It
    # is what PySlice's WFData.counts() intends -- a fixed total spread over
    # the pattern by its probability -- written here because theirs cannot
    # be run on a realistic diffraction pattern (see Notes). Nothing in the
    # pipeline should call this to generate data.
    """
    Draw multinomial counts: the reference wrong answer.

    Parameters
    ----------
    lam : numpy.ndarray
        Expected counts per pixel. Its sum, rounded, becomes the fixed total.
    rng : numpy.random.Generator
        Generator from :func:`make_rng`.

    Returns
    -------
    numpy.ndarray
        Integer counts summing to exactly ``round(lam.sum())`` every draw.

    Notes
    -----
    Kept for :func:`sample_counts` to be tested against. Multinomial differs
    from Poisson in three ways that matter here: the total is fixed, so the
    dose carries no shot noise of its own; pixels are weakly anti-correlated
    rather than independent; and PySlice normalises over the whole
    ``(probe, frame, kx, ky, layer)`` array, which for a scan would couple
    probe positions to each other.

    PySlice's implementation also cannot run on a cropped diffraction
    pattern. It builds histogram bins from ``cumsum`` of the probabilities,
    and a pattern with large dark regions produces a cumulative sum that is
    not monotonic under floating-point rounding -- 41 backward steps of
    about 1e-16 in 24843 pixels, on the torch backend in float64 -- so
    ``numpy.histogram`` raises ``ValueError: bins must increase
    monotonically``. Accumulating a running maximum over the bins would fix
    it.
    """
    total = int(round(float(lam.sum())))
    probability = np.asarray(lam, dtype=float).ravel()
    probability = probability / probability.sum()
    return rng.multinomial(total, probability).reshape(lam.shape)
