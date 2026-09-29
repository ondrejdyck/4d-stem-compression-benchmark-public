"""Streaming a scan: patterns produced, sampled, handed on, discarded.

The scan is streamed rather than materialised. The cube is the thing the
manuscript is about, and holding all of it at once would put the simulation
beyond the machines that have to run it.

This is not only a matter of principle. PySlice's ``run()`` allocates the
full ``(n_probes, n_frames, nkx, nky, n_layers)`` complex array up front,
which at 16 frozen-phonon configurations is **6.07 MiB per probe position**
-- a 256x256 scan would want 388 GiB. Batching is therefore mandatory, and
the batch size is the memory knob.

Measured cost, this grid and 16 configurations: ~0.3 s of fixed
propagation per batch plus ~40 ms per probe position. A batch of 256
carries 3% overhead, a batch of 16 carries 30%.
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple, Optional, Sequence

import numpy as np

from paper_artifacts.simulation import config
from paper_artifacts.simulation.diffraction import simulate
from paper_artifacts.simulation.dose import collected_fraction, expected_counts, sample_counts
from paper_artifacts.simulation.potential import GridSpec
from paper_artifacts.simulation.specimen import thermal_trajectory


class ScanPoint(NamedTuple):
    """One probe position's worth of data, as the model would receive it.

    Attributes
    ----------
    index : int
        Position index in scan order.
    position : tuple of float
        Probe centre in angstroms.
    counts : numpy.ndarray
        ``(n_scans, nkx, nky)`` integer counts, Poisson given lambda.
    collected_fraction : float
        Fraction of the incident beam still represented on the detector
        grid. Below 1 where the specimen scatters past the multislice
        anti-aliasing aperture. A diagnostic, not an observable; see
        :func:`paper_artifacts.simulation.dose.collected_fraction`.
    theta_rad, theta_x_rad, theta_y_rad : numpy.ndarray
        Detector geometry, shared references rather than copies. Carried on
        every point so that a consumer taking points off a queue is
        self-sufficient, which is what a stream should be.
    lam : numpy.ndarray or None
        Noiseless expected counts. ``None`` unless explicitly requested.
        **This is ground truth.** Anything that gives it to the model is
        not running the experiment.
    """

    index: int
    position: tuple[float, float]
    counts: np.ndarray
    collected_fraction: float
    theta_rad: np.ndarray
    theta_x_rad: np.ndarray
    theta_y_rad: np.ndarray
    lam: Optional[np.ndarray] = None


def scan_positions(
    spec: GridSpec,
    n_x: int,
    n_y: int,
    origin_A: Optional[tuple[float, float]] = None,
    extent_A: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    # Why this exists: probe positions are the one input where an off-by-one
    # or an inclusive endpoint silently changes the specimen being scanned --
    # sampling x=0 and x=L twice under periodic boundaries duplicates a
    # column of the image and shifts every lattice measurement made from it.
    """
    Raster of probe positions over the cell or a region of it.

    Parameters
    ----------
    spec : GridSpec
        Grid from :func:`paper_artifacts.simulation.potential.grid_spec`.
    n_x, n_y : int
        Positions along each axis.
    origin_A : tuple of float, optional
        Lower-left corner in angstroms. Default the cell origin.
    extent_A : tuple of float, optional
        Region size in angstroms. Default the whole cell.

    Returns
    -------
    numpy.ndarray
        ``(n_x * n_y, 2)`` positions in row-major scan order.

    Notes
    -----
    Endpoints are excluded. The cell is periodic, so including both ends
    would scan the same column twice.
    """
    origin = (0.0, 0.0) if origin_A is None else origin_A
    if extent_A is None:
        extent_A = (
            float(spec.xs[-1] + spec.sampling_x_A),
            float(spec.ys[-1] + spec.sampling_y_A),
        )
    xs = origin[0] + np.linspace(0.0, extent_A[0], n_x, endpoint=False)
    ys = origin[1] + np.linspace(0.0, extent_A[1], n_y, endpoint=False)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def stream_scan(
    trajectory: Any,
    spec: GridSpec,
    positions: Sequence[tuple[float, float]],
    rng: np.random.Generator,
    n_scans: int = 1,
    batch_size: int = 64,
    n_electrons: Optional[float] = None,
    n_configurations: int = config.FROZEN_PHONON_CONFIGS,
    sigma_A: float = config.FROZEN_PHONON_SIGMA_A,
    seed: int = config.FROZEN_PHONON_SEED,
    include_lambda: bool = False,
) -> Iterator[ScanPoint]:
    # Why this exists: to make the memory profile of the simulator a property
    # of the batch size rather than of the scan, and to make the specimen
    # fixed across the whole scan by construction. The displaced trajectory
    # is built once, before the loop. Building it inside would redraw the
    # thermal state per batch, so every position would sample a different
    # specimen -- not a noisier experiment but a different one, and it would
    # surface much later as unexplained model error.
    """
    Yield one probe position at a time, with counts drawn as they go.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Static single-frame specimen. Displacements are applied here.
    spec : GridSpec
        Grid from :func:`paper_artifacts.simulation.potential.grid_spec`.
    positions : sequence of (float, float)
        Probe positions in angstroms, in scan order.
    rng : numpy.random.Generator
        Generator for the counts, from :func:`paper_artifacts.simulation.dose.make_rng`.
    n_scans : int, optional
        Repeat passes over each position, drawn independently from the same
        lambda. Default 1.
    batch_size : int, optional
        Probe positions propagated per call. The memory knob: peak usage is
        about ``6 MiB * batch_size`` at 16 configurations. Default 64.
    n_electrons : float, optional
        Dose per position. Default :func:`paper_artifacts.simulation.dose.electrons_per_probe`.
    n_configurations, sigma_A, seed : optional
        Frozen-phonon state. Defaults from :mod:`paper_artifacts.simulation.config`. Together
        these define the specimen; see :func:`paper_artifacts.simulation.specimen.thermal_trajectory`.
    include_lambda : bool, optional
        Attach the noiseless expected counts to each point. **Ground truth,
        for Phase H only.** Default False.

    Yields
    ------
    ScanPoint
        One per position, in scan order.

    Notes
    -----
    Each batch re-propagates the specimen, costing ~0.3 s on this grid, so
    very small batches are wasteful rather than wrong. The counts of one
    position do not depend on the batching; only memory does.
    """
    displaced = thermal_trajectory(
        trajectory,
        n_configurations=n_configurations,
        sigma_A=sigma_A,
        seed=seed,
    )
    positions = np.asarray(positions, dtype=float)

    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        diff = simulate(
            displaced,
            spec,
            positions=[tuple(p) for p in batch],
            n_configurations=n_configurations,
            sigma_A=sigma_A,
            seed=seed,
            displace=False,
        )

        for offset in range(len(batch)):
            lam = expected_counts(diff, n_electrons=n_electrons, position=offset)
            counts = np.stack([sample_counts(lam, rng) for _ in range(n_scans)])
            yield ScanPoint(
                index=start + offset,
                position=(float(batch[offset, 0]), float(batch[offset, 1])),
                counts=counts,
                collected_fraction=collected_fraction(diff, position=offset),
                theta_rad=diff.theta_rad,
                theta_x_rad=diff.theta_x_rad,
                theta_y_rad=diff.theta_y_rad,
                lam=lam if include_lambda else None,
            )
