"""Multislice grid and projected potential for a specimen.

The grid is where most silent errors enter a multislice simulation: too coarse
a sampling and the detector quietly extends past the largest representable
angle, so the high-angle signal is wrong with no array shape changing and no
exception raised. This module derives the grid from the rules in
:mod:`paper_artifacts.simulation.config` and reports what it actually achieved, so a
caller can check the realised values rather than trust the intended ones.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from pyslice.multislice.potentials import Potential, grid_from_trajectory

from paper_artifacts.simulation import config


class GridSpec(NamedTuple):
    """Realised multislice grid, as opposed to the requested one.

    Attributes
    ----------
    xs, ys, zs : numpy.ndarray
        Coordinate axes in angstroms.
    sampling_x_A, sampling_y_A : float
        Achieved real-space sampling per axis.
    requested_sampling_A : float
        Sampling *asked* of PySlice. This is the value to pass back to
        ``MultisliceCalculator.setup``, which re-derives the same grid from
        it. Kept on the spec so a caller cannot build a grid at one sampling
        and then silently propagate at another.
    slice_thickness_A : float
        Slice thickness *requested* of PySlice. This is the value to pass
        back to ``MultisliceCalculator.setup``; it is not the spacing the
        grid ends up with.
    realised_slice_thickness_A : float
        Spacing the grid actually has, ``zs[1] - zs[0]``. PySlice computes
        ``nz = int(lz / slice_thickness) + 1`` and then spreads ``nz``
        points evenly over the cell, so asking for 0.489 Å over a 7.34 Å
        cell yields 16 slices of 0.459 Å rather than 15 of 0.489. Both are
        fine physically; recording the requested number as though it were
        the achieved one is not.
    n_slices : int
        Number of propagation slices the grid actually has, ``len(zs)``.
    theta_max_rad : float
        Largest scattering angle the grid represents, from the multislice
        band limit ``k <= 1/(3*sampling)``.
    """

    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    sampling_x_A: float
    sampling_y_A: float
    requested_sampling_A: float
    slice_thickness_A: float
    realised_slice_thickness_A: float
    n_slices: int
    theta_max_rad: float


def grid_spec(
    trajectory: Any,
    sampling_A: float = config.SAMPLING_A,
    slice_target_A: float = config.SLICE_THICKNESS_TARGET_A,
    wavelength_A: float = config.WAVELENGTH_A,
) -> GridSpec:
    # Why this exists: PySlice's grid_from_trajectory takes a requested
    # sampling and returns axes, but the realised sampling differs because the
    # cell must divide into a whole number of samples. Every downstream angle
    # -- the aperture, the ADF annulus, the detector calibration -- depends on
    # what was realised, not what was asked for. Computing theta_max here, from
    # the axes that actually came back, is what lets C-T3 be a real check
    # rather than a restatement of our own intent.
    """
    Build the multislice grid and report what it achieved.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Specimen with a diagonal box.
    sampling_A : float, optional
        Requested real-space sampling. Default :data:`paper_artifacts.simulation.config.SAMPLING_A`.
    slice_target_A : float, optional
        Target slice thickness; adjusted to divide the cell height evenly
        per rule R3. Default :data:`paper_artifacts.simulation.config.SLICE_THICKNESS_TARGET_A`.
    wavelength_A : float, optional
        Electron wavelength, used to convert the band limit to an angle.
        Default :data:`paper_artifacts.simulation.config.WAVELENGTH_A`.

    Returns
    -------
    GridSpec
        Realised grid, sampling, slicing and maximum representable angle.
    """
    height = float(np.asarray(trajectory.box_matrix)[2, 2])
    n_slices = max(1, round(height / slice_target_A))
    slice_thickness = height / n_slices

    xs, ys, zs, _, _, _ = grid_from_trajectory(
        trajectory, sampling=sampling_A, slice_thickness=slice_thickness
    )
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    # Multislice band limit: the anti-aliasing aperture keeps k <= 1/(3*dx),
    # so the largest fully represented angle is lambda/(3*dx). The coarser
    # axis is the binding one.
    theta_max = wavelength_A / (3.0 * max(dx, dy))

    return GridSpec(
        xs=xs,
        ys=ys,
        zs=zs,
        sampling_x_A=dx,
        sampling_y_A=dy,
        requested_sampling_A=sampling_A,
        slice_thickness_A=slice_thickness,
        realised_slice_thickness_A=float(zs[1] - zs[0]),
        n_slices=len(zs),
        theta_max_rad=theta_max,
    )


def build_potential(
    trajectory: Any,
    spec: GridSpec,
    frame: int = 0,
    project: bool = True,
) -> np.ndarray:
    # Why this exists: the projected potential is the only place the specimen
    # enters the simulation, so it is the right thing to test the structure
    # against. Its Fourier transform must show the WSe2 reciprocal lattice; if
    # it does not, either the specimen or the grid is wrong, and every
    # diffraction pattern computed afterwards would be wrong in a way that
    # still looks like a diffraction pattern.
    """
    Compute the Kirkland potential for a specimen on a given grid.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Specimen.
    spec : GridSpec
        Grid from :func:`grid_spec`.
    frame : int, optional
        Frame index. Default 0.
    project : bool, optional
        When True, collapse the slice axis to a single projected plane, which
        is the form the lattice test wants. When False, return the full
        ``(nx, ny, n_slices)`` array used for propagation. Default True.

    Returns
    -------
    numpy.ndarray
        Projected potential ``(nx, ny)`` when ``project``, otherwise the
        full sliced array.
    """
    positions = np.asarray(trajectory.positions)[frame]
    potential = Potential(
        spec.xs,
        spec.ys,
        spec.zs,
        positions=positions,
        atom_types=trajectory.atom_types,
        kind="kirkland",
        slice_axis=2,
    )
    potential.build()
    if project:
        potential.flatten()
        return potential.to_numpy()[:, :, 0]
    return potential.to_numpy()
