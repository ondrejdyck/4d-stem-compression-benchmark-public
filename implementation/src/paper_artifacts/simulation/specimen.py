"""Specimen construction.

The specimen is the ground truth. Everything downstream -- the projected
potential, the simulated diffraction, the Poisson counts -- is judged against
what goes in here, so this module is deliberately small.
"""

from __future__ import annotations

from typing import Any, Literal

from ase.build import mx2
from pyslice.io.build import build_slab

# Literature values for monolayer 2H-WSe2. Reported in-plane constants cluster
# around 3.28-3.30 A; the Se-Se separation is the sheet's full thickness.
WSE2_LATTICE_A: float = 3.28
WSE2_SE_SE_A: float = 3.34

# Vacuum added along the beam, per side, by ``mx2``. Two constraints pull in
# opposite directions. Too little and the sheet's periodic images touch: with
# zero vacuum the cell is exactly the 3.34 A sheet thickness, so the Se plane
# at z=0 and the one at z=3.34 are the SAME position under PBC and the two
# chalcogen planes silently merge into one. Too much and we pay propagation
# steps for empty space, because PySlice slices the cell height rather than
# the atom extent: 2 A per side gives 15 slices where 6 A would give 43.
DEFAULT_VACUUM_A: float = 2.0


def build_wse2_monolayer(
    min_lateral_A: float = 20.0,
    vacuum_A: float = DEFAULT_VACUUM_A,
    lattice_a_A: float = WSE2_LATTICE_A,
    thickness_A: float = WSE2_SE_SE_A,
    kind: Literal["2H", "1T"] = "2H",
) -> tuple[Any, dict[str, Any]]:
    # Why this exists: every later phase needs a specimen whose structure we
    # know exactly, in the one form PySlice's multislice grid accepts -- a
    # single-frame Trajectory with a diagonal box that is exactly periodic in
    # plane. ASE's mx2 gives the right chemistry on a hexagonal cell; PySlice's
    # build_slab is what makes it orthogonal, periodic, and diagonal. Going
    # through ASE alone would leave a non-periodic edge that shows up later as
    # a seam in the projected potential and spurious diffraction, which is
    # exactly the kind of error that produces a plausible-looking wrong answer.
    """
    Build an exactly periodic WSe2 monolayer as a PySlice ``Trajectory``.

    Parameters
    ----------
    min_lateral_A : float, optional
        Minimum in-plane extent in angstroms. Rounded up to whole cell
        repeats by :func:`pyslice.io.build.build_slab`. Must exceed four
        probe diameters or periodic images of the probe overlap; see rule R5
        of PySlice's ``simulation-parameter-selection`` skill. Default 20.0.
    vacuum_A : float, optional
        Vacuum per side along the beam axis, applied when building the unit
        cell. Must be strictly positive or the two Se planes coincide under
        periodic boundaries. Default :data:`DEFAULT_VACUUM_A`.
    lattice_a_A : float, optional
        In-plane lattice constant. Default :data:`WSE2_LATTICE_A`.
    thickness_A : float, optional
        Se-Se separation, the sheet's full thickness.
        Default :data:`WSE2_SE_SE_A`.
    kind : {"2H", "1T"}, optional
        Transition-metal dichalcogenide polytype. Default ``"2H"``.

    Returns
    -------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Single-frame trajectory with a diagonal box, ready for
        ``MultisliceCalculator.setup``.
    record : dict
        Build record from ``build_slab``: orthogonalizing supercell, lateral
        repeats, final box, atom count. Persist this -- it is the provenance
        of the ground truth.

    Notes
    -----
    All vacuum is applied by ``mx2``; ``build_slab`` is called with
    ``vacuum_A=0.0``. Adding it in both places double-counts and inflates the
    slice count. ``vacuum=None`` is not usable -- it leaves a zero-height
    cell that trips an assertion inside ASE's surface builder -- and
    ``vacuum=0.0`` silently merges the two Se planes (see above), so this
    argument must stay strictly positive.

    Examples
    --------
    >>> traj, record = build_wse2_monolayer(min_lateral_A=20.0)
    >>> record["n_atoms"] % 3 == 0
    True
    """
    if vacuum_A <= 0.0:
        raise ValueError(
            "vacuum_A must be > 0: with zero vacuum the cell height equals the "
            "sheet thickness, so the two Se planes coincide under periodic "
            f"boundaries and merge into one. Got {vacuum_A}."
        )
    unit_cell = mx2(
        formula="WSe2",
        kind=kind,
        a=lattice_a_A,
        thickness=thickness_A,
        size=(1, 1, 1),
        vacuum=vacuum_A,
    )
    trajectory, record = build_slab(
        unit_cell,
        indices=(0, 0, 1),
        layers=1,
        min_lateral_A=min_lateral_A,
        vacuum_A=0.0,
    )
    return trajectory, record


def thermal_trajectory(
    trajectory: Any,
    n_configurations: int,
    sigma_A: float,
    seed: int,
) -> Any:
    # Why this exists: the frozen thermal state is part of the specimen, not
    # a property of the propagation, so it belongs here beside the atoms. It
    # is a thin wrapper over PySlice's generator for two reasons. First, the
    # seed is mandatory: a fresh draw each call would make the ground truth
    # undefined, and an unseeded default is the easiest possible way to get
    # that by accident. Second, it is the single place that documents the
    # sigma/sqrt(3) per-component convention, which is a 1.7x error in every
    # displacement if applied twice or not at all, and invisible in any image.
    """
    Draw a fixed set of frozen-phonon configurations.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Single-frame specimen from :func:`build_wse2_monolayer`.
    n_configurations : int
        Number of displaced configurations. Each becomes one frame.
    sigma_A : float
        RMS displacement *magnitude* in angstroms, applied to every atom
        regardless of species.
    seed : int
        Seed for the displacement draw. Required, not optional.

    Returns
    -------
    pyslice.multislice.trajectory.Trajectory
        Multi-frame trajectory, one frame per configuration.

    Raises
    ------
    ValueError
        If ``seed`` is None, or if the input has more than one frame.

    Notes
    -----
    PySlice draws each Cartesian component from ``N(0, sigma/sqrt(3))``, so
    ``sigma_A`` is the RMS magnitude ``sqrt(<|d|^2>)`` and the per-axis RMS
    is ``sigma_A/sqrt(3)``.

    The configurations are fixed once drawn and are treated as part of the
    specimen. That is deliberate: it leaves shot noise as the only stochastic
    element in the generated data, which is exactly what the model must see
    through. Re-drawing per scan would add a second noise source and confound
    the experiment.

    Examples
    --------
    >>> traj, _ = build_wse2_monolayer(min_lateral_A=17.0)
    >>> thermal_trajectory(traj, 4, 0.09, seed=0).n_frames
    4
    """
    if seed is None:
        raise ValueError(
            "seed is required: the displaced configurations are part of the "
            "ground-truth definition, so an unseeded draw would leave the "
            "specimen undefined."
        )
    if trajectory.n_frames != 1:
        raise ValueError(
            "expected a single-frame trajectory to displace, got "
            f"{trajectory.n_frames} frames; displacing an existing thermal "
            "state would compound two sets of displacements."
        )
    return trajectory.generate_random_displacements(
        n_displacements=n_configurations, sigma=sigma_A, seed=seed
    )
