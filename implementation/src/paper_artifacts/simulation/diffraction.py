"""Multislice propagation and the resulting diffraction patterns.

This is the last deterministic stage: everything here is exact physics, and
the Poisson sampling in :mod:`paper_artifacts.simulation.dose`
sits on top of it. The functions return the
angular axis alongside the intensity so no caller has to reconstruct the
calibration, which is the step most likely to be got wrong silently.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
from pyslice.backend import to_numpy
from pyslice.multislice.calculators import MultisliceCalculator

from paper_artifacts.simulation import config
from paper_artifacts.simulation.potential import GridSpec
from paper_artifacts.simulation.specimen import thermal_trajectory


class Diffraction(NamedTuple):
    """Diffraction patterns with their angular calibration.

    Attributes
    ----------
    intensity : numpy.ndarray
        ``(n_positions, nkx, nky)`` diffracted intensity, ``|psi_k|^2``,
        averaged incoherently over frozen-phonon configurations.
    theta_rad : numpy.ndarray
        ``(nkx, nky)`` scattering angle of each detector pixel, radians.
    theta_x_rad, theta_y_rad : numpy.ndarray
        Signed components of the same angle. ``theta_rad`` is their
        hypotenuse; both are kept because a centre-of-mass observable needs
        the sign and cannot recover it.
    incident_total : float
        Total incident intensity in reciprocal space, for flux accounting.
    n_configurations : int
        Frozen-phonon configurations averaged. 1 means a static specimen.
    sigma_A : float
        RMS displacement magnitude used, angstroms. 0.0 when static.
    seed : int or None
        Seed of the displacement draw. None when static.

    Notes
    -----
    The last three fields are provenance, and they are load-bearing. The
    seeded configuration set is part of the specimen, so these numbers define
    the ground truth rather than merely recording how it was made.
    """

    intensity: np.ndarray
    theta_rad: np.ndarray
    theta_x_rad: np.ndarray
    theta_y_rad: np.ndarray
    incident_total: float
    n_configurations: int = 1
    sigma_A: float = 0.0
    seed: Optional[int] = None


def _propagate(
    trajectory: Any,
    spec: GridSpec,
    positions: Sequence[tuple[float, float]],
    slice_thickness_A: Optional[float] = None,
    crop_to_theta_max: bool = True,
) -> tuple[MultisliceCalculator, Any]:
    # Why this exists: it is the one place that turns the grid built in
    # potential.py into MultisliceCalculator.setup arguments, so the grid that
    # was checked and the grid the physics ran on cannot drift apart. It is
    # separate from simulate() only so that a caller can reach the raw
    # WFData, to compare PySlice's own count sampler against ours.
    """
    Run the multislice and return the calculator and its wavefunction.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Specimen, one frame per frozen-phonon configuration.
    spec : GridSpec
        Grid from :func:`paper_artifacts.simulation.potential.grid_spec`.
    positions : sequence of (float, float)
        Probe positions in angstroms.
    slice_thickness_A : float, optional
        Overrides the grid's requested slice thickness.
    crop_to_theta_max : bool, optional
        Crop reciprocal space to :data:`paper_artifacts.simulation.config.THETA_MAX_RAD`.

    Returns
    -------
    calculator : pyslice.multislice.calculators.MultisliceCalculator
        Kept for its ``base_probe``, which carries the incident intensity.
    wf : pyslice.postprocessing.wf_data.WFData
        Complex array shaped ``(probe, frame, kx, ky, layer)``.
    """
    kwargs: dict[str, Any] = dict(
        aperture=config.APERTURE_RAD * 1e3,
        voltage_eV=config.VOLTAGE_V,
        slice_thickness=slice_thickness_A or spec.slice_thickness_A,
        sampling=spec.requested_sampling_A,
        probe_positions=list(positions),
        save_path=config.cache_root(),
        cache_wavefunctions=False,
    )
    if crop_to_theta_max:
        k_max = config.THETA_MAX_RAD / config.WAVELENGTH_A
        kwargs.update(max_kx=k_max, max_ky=k_max)

    calculator = MultisliceCalculator()
    calculator.setup(trajectory, **kwargs)
    result = calculator.run()
    wf = result[1] if isinstance(result, (list, tuple)) else result
    return calculator, wf


def simulate(
    trajectory: Any,
    spec: GridSpec,
    positions: Optional[Sequence[tuple[float, float]]] = None,
    slice_thickness_A: Optional[float] = None,
    crop_to_theta_max: bool = True,
    n_configurations: Optional[int] = None,
    sigma_A: Optional[float] = None,
    seed: Optional[int] = None,
    displace: bool = True,
) -> Diffraction:
    # Why this exists: MultisliceCalculator.setup takes sampling and slice
    # thickness and builds its own grid, so it can silently disagree with the
    # GridSpec every test was written against. Funnelling every call through
    # here, with the same numbers, is what keeps the grid the tests verified
    # and the grid the physics ran on the same grid.
    """
    Propagate a probe through a specimen and return diffraction patterns.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Specimen.
    spec : GridSpec
        Grid from :func:`paper_artifacts.simulation.potential.grid_spec`.
    positions : sequence of (float, float), optional
        Probe positions in angstroms. Defaults to the cell centre.
    slice_thickness_A : float, optional
        Overrides the grid's slice thickness. Used by the convergence test.
    crop_to_theta_max : bool, optional
        Crop reciprocal space to :data:`paper_artifacts.simulation.config.THETA_MAX_RAD`.
        Set False for flux accounting, since cropping discards scatter.
        Default True.
    n_configurations : int, optional
        Frozen-phonon configurations to average over. Default None, meaning
        a static specimen. Not required for a dark-field observable -- see
        Notes -- but it is part of the ground-truth definition, so a
        multi-frame trajectory must declare it.
    sigma_A : float, optional
        RMS displacement magnitude in angstroms. Required when
        ``n_configurations`` is given.
    seed : int, optional
        Seed for the displacement draw. Required when ``n_configurations``
        is given; it is part of the ground-truth definition.
    displace : bool, optional
        When True (default) the displacements are drawn here. Pass False to
        declare the provenance of a trajectory that is *already* displaced,
        as :func:`paper_artifacts.simulation.scan.stream_scan` does when it draws once for
        a whole scan. The frame count is then checked against
        ``n_configurations``.

    Returns
    -------
    Diffraction
        Patterns, angular axis, incident total, and the thermal provenance.

    Notes
    -----
    Configurations are averaged in *intensity*, not amplitude. Averaging
    amplitudes would be a coherent sum and would reproduce the static
    result exactly -- silent, and indistinguishable from a static run at
    every later stage.

    Displacements are not required for a converged ADF observable. Static
    ADF at a tungsten column is 3.6e-2 of the beam and converges to 0.13%
    under slice refinement. What displacements add is thermal diffuse
    scattering, which moves the ADF fraction by 9-18% depending on site and
    compresses the W/Se contrast ratio from 1.96 to 1.62. That change to the
    observable is why this path exists.
    """
    if positions is None:
        positions = [(float(spec.xs.mean()), float(spec.ys.mean()))]

    if n_configurations is None:
        # A multi-frame trajectory here would be averaged correctly but
        # recorded as static, so the provenance on the result would be a
        # lie about the specimen that produced it.
        if trajectory.n_frames != 1:
            raise ValueError(
                f"trajectory has {trajectory.n_frames} frames but no thermal "
                "provenance was declared. Pass n_configurations, sigma_A and "
                "seed with displace=False."
            )
        n_configurations, sigma_A = 1, 0.0
    elif displace:
        if sigma_A is None:
            raise ValueError("sigma_A is required when n_configurations is given")
        trajectory = thermal_trajectory(
            trajectory, n_configurations=n_configurations, sigma_A=sigma_A, seed=seed
        )
    elif trajectory.n_frames != n_configurations:
        raise ValueError(
            f"trajectory has {trajectory.n_frames} frames but "
            f"n_configurations={n_configurations} was declared"
        )

    calculator, wf = _propagate(
        trajectory,
        spec,
        positions,
        slice_thickness_A=slice_thickness_A,
        crop_to_theta_max=crop_to_theta_max,
    )

    array = to_numpy(wf.array)
    intensity = (np.abs(array[:, :, :, :, 0]) ** 2).mean(axis=1)

    kx = to_numpy(wf.kxs)
    ky = to_numpy(wf.kys)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    theta_x = kxx * config.WAVELENGTH_A
    theta_y = kyy * config.WAVELENGTH_A
    theta = np.hypot(theta_x, theta_y)

    # Accumulate the normalisation in float64 whatever the propagation ran
    # in. This is a sum over ~56000 pixels that every count in the dataset is
    # scaled by, so it is the one place single precision would actually cost
    # something -- and it is cheap here, being one small CPU transform.
    probe = to_numpy(calculator.base_probe._array)[0, 0].astype(np.complex128)
    incident = np.abs(np.fft.fft2(probe)) ** 2

    return Diffraction(
        intensity=intensity,
        theta_rad=theta,
        theta_x_rad=theta_x,
        theta_y_rad=theta_y,
        incident_total=float(incident.sum()),
        n_configurations=n_configurations,
        sigma_A=float(sigma_A),
        seed=seed,
    )


def detector_fraction(diff: Diffraction, inner_rad: float, outer_rad: float, position: int = 0) -> float:
    # Why this exists: every observable here is an integral of the
    # diffraction pattern over an angular range, bright field and annular dark
    # field alike. Having one place that applies an annular mask
    # means the angular convention is defined once and cannot drift between
    # the ground truth and the thing being compared against it.
    """
    Fraction of diffracted intensity falling in an annular detector.

    Parameters
    ----------
    diff : Diffraction
        Patterns from :func:`simulate`.
    inner_rad, outer_rad : float
        Detector angles in radians. ``inner_rad=0`` gives a disk.
    position : int, optional
        Probe-position index. Default 0.

    Returns
    -------
    float
        Collected intensity divided by the total in the pattern.
    """
    if outer_rad > 0.75 * config.THETA_MAX_RAD:
        warnings.warn(
            f"detector outer angle {outer_rad*1e3:.0f} mrad is more than 0.6 of "
            f"the {config.THETA_MAX_RAD*1e3:.0f} mrad band limit. The "
            "anti-aliasing roll-off is applied at every slice, so an integral "
            "out here drifts with real-space sampling: the 90-150 mrad "
            "detector moved 24% between 0.077 and 0.040 A at a tungsten "
            "column. Refine sampling before quoting it.",
            UserWarning,
            stacklevel=2,
        )
    intensity = np.asarray(diff.intensity[position], dtype=np.float64)
    mask = (diff.theta_rad >= inner_rad) & (diff.theta_rad <= outer_rad)
    return float(intensity[mask].sum() / intensity.sum())
