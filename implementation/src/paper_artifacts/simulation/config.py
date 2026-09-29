"""Physical and numerical parameters for the simulation.

Every value here is either chosen deliberately or derived from PySlice's
documented parameter-selection rules. The rule number is given for each
derived value. Do not re-derive these independently; if a rule is wrong, it is
wrong in one place.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# --- chosen ---------------------------------------------------------------

#: Accelerating voltage (V). 80 kV is usual for 2D materials: high enough for
#: resolution, low enough to limit knock-on damage.
VOLTAGE_V: float = 80_000.0

#: Convergence semiangle (rad). 30 mrad is a typical aberration-corrected
#: STEM probe (R7).
APERTURE_RAD: float = 30e-3

#: Largest scattering angle the grid represents (rad). Deliberately NOT
#: re-derived from the ADF outer angle. R10's 1.2x is a *representability*
#: margin -- enough that the detector is not sitting in the anti-aliasing
#: taper -- and it was being used here as though it were a *convergence*
#: margin, which it is not. SAMPLING_A = lambda/(3*THETA_MAX_RAD) puts the
#: multislice anti-aliasing cutoff at exactly this angle, and that cutoff is
#: a soft roll-off applied at every slice, so an observable in the outer half
#: of the range drifts as sampling changes. Holding this at 180 mrad while
#: the detector sits at 75-130 gives a margin of 1.4x on the outer angle.
THETA_MAX_RAD: float = 180e-3

#: ADF outer angle (rad). 0.72 of the band limit; see THETA_MAX_RAD.
ADF_OUTER_RAD: float = 130e-3

#: ADF inner angle (rad). 2.5x the aperture.
#:
#: This is a compromise between two things that pull opposite ways. Pushing
#: the detector out sharpens Z-contrast, which is a high-angle phenomenon;
#: pushing it in keeps it clear of the anti-aliasing roll-off, which makes
#: the observable drift with sampling. Measured W/Se2 column ratio and the
#: drift between 0.077 and 0.040 A sampling at a tungsten column:
#:
#:     60-100 mrad : W/Se2 1.05, drift 10%   -- almost no Z-contrast
#:     75-130 mrad : W/Se2 1.40, drift 18%   <- chosen
#:     90-150 mrad : W/Se2 1.40, drift 24%
#:
#: 75-130 buys the full converged Z-contrast for a third less grid
#: sensitivity than 90-150. Note 1.40 is the *converged* ratio; on the
#: default grid it reads 1.59, so the contrast itself is ~14% inflated.
ADF_INNER_RAD: float = 75e-3

# --- frozen phonons -------------------------------------------------------
# These three numbers, together, *define* the ground truth. Since the
# simulation is the ground truth by definition, they are not attempts to
# reproduce WSe2's true thermal state -- they are a statement of what we put
# in. Changing any of them changes the specimen the model is asked to recover.
#
# Static ADF is real, converged scattering; the observable does not depend on
# displacements. They are kept because frozen-phonon averaging is standard
# multislice practice and because a fixed, seeded thermal state costs nothing
# to define. At 75-130 mrad they shift the ADF fraction by a few percent.

#: RMS displacement magnitude (Å) applied to every atom. In the usual
#: 0.05-0.1 Å range; not matched to a literature Debye-Waller factor,
#: which would buy nothing when the simulation is the ground truth.
#: PySlice applies one sigma to all species; for WSe2 that under-displaces
#: Se relative to W, a knowing approximation with no consequence here.
FROZEN_PHONON_SIGMA_A: float = 0.09

#: Number of displaced configurations averaged incoherently. Chosen so the
#: ground truth is stable, not to converge a real thermal average.
FROZEN_PHONON_CONFIGS: int = 16

#: Seed for the displacement draw. Part of the ground-truth definition.
FROZEN_PHONON_SEED: int = 20260916

# --- dose -----------------------------------------------------------------

#: Elementary charge (C). SI 2019 exact value.
ELEMENTARY_CHARGE_C: float = 1.602176634e-19

#: Beam current (A). 60 pA is roughly the current used for the 4D_Diff
#: dataset in the companion compression-benchmark paper, so signal-to-noise
#: statements here can be read against measurements there.
BEAM_CURRENT_A: float = 60e-12

#: Dwell time per probe position (s).
DWELL_S: float = 1e-6

# --- numerics -------------------------------------------------------------

#: Multislice arithmetic precision, "single" or "double". Overridable with
#: ``$PLAIN_PRECISION``.
#:
#: PySlice defaults to double on CUDA. On the GPU used here that is 3.8x
#: slower end to end (46.3 against 12.2 ms per probe position) and buys
#: nothing measurable: the ADF fraction moves by 1.5e-7 relative and lambda
#: by 9e-6 of peak, both far below the 3% frozen-phonon seed spread already
#: present in the simulation. :mod:`paper_artifacts.simulation._precision`
#: records the measurements.
MULTISLICE_PRECISION: str = os.environ.get("PLAIN_PRECISION", "single")


def wavelength_A(voltage_V: float = VOLTAGE_V) -> float:
    # Why this exists: every sampling rule below is a ratio of the wavelength
    # to an angle, so a wrong wavelength silently mis-scales the entire grid
    # without any array shape changing. Keeping the relativistic formula in
    # one named place makes that failure impossible to introduce twice.
    """
    Relativistic electron wavelength.

    Parameters
    ----------
    voltage_V : float, optional
        Accelerating voltage in volts. Default :data:`VOLTAGE_V`.

    Returns
    -------
    float
        Wavelength in angstroms.

    Notes
    -----
    R1 of PySlice's parameter-selection skill:
    ``lambda = 12.2639 / sqrt(V + 0.97845e-6 * V**2)``.

    Examples
    --------
    >>> round(wavelength_A(80_000.0), 5)
    0.04176
    """
    return 12.2639 / math.sqrt(voltage_V + 0.97845e-6 * voltage_V**2)


#: Electron wavelength at :data:`VOLTAGE_V` (Å).
WAVELENGTH_A: float = wavelength_A()

# --- derived --------------------------------------------------------------

#: Real-space sampling (Å). R2: the multislice band limit keeps k <= 1/(3*s),
#: so representing scattering to THETA_MAX_RAD needs s <= lambda/(3*theta).
SAMPLING_A: float = WAVELENGTH_A / (3.0 * THETA_MAX_RAD)

#: Slice thickness target (Å). R3 wants ~0.5 Å, adjusted to divide the cell
#: height evenly; the exact value is computed per specimen.
SLICE_THICKNESS_TARGET_A: float = 0.5

#: Probe diameter (Å). R5: d ~ 1.22 * lambda / alpha.
PROBE_DIAMETER_A: float = 1.22 * WAVELENGTH_A / APERTURE_RAD

#: Minimum lateral cell extent (Å). R4 wants >= 5*lambda/alpha for k-space
#: sampling of the aperture disk; R5 wants >= 4 probe diameters to stop the
#: probe overlapping its periodic image. 17 Å clears both by a wide margin and
#: is the largest cell whose grid still lands at 256 samples for the chosen
#: SAMPLING_A -- a smaller cell would oversample, a larger one would either
#: undersample or cost pixels we do not need.
MIN_LATERAL_A: float = 17.0

#: Probe step (Å). R11 refines R6 for atomic resolution: d1/(2*oversample)
#: with oversample ~10, where d1 is the widest in-plane lattice spacing.
#: Computed per specimen from ``first_bragg_g``; this is the fallback.
PROBE_STEP_A: float = 0.142


def cache_root() -> Path:
    # Why this exists: PySlice caches wavefunctions to ``psi_data/`` in the
    # current working directory, which puts many gigabytes of regenerable
    # intermediates wherever the command happened to be run -- inside a synced
    # folder, for instance. Everything under here is reproducible from a seed,
    # so a scratch directory is the right home for it.
    """
    Directory for PySlice's wavefunction cache.

    Returns
    -------
    pathlib.Path
        ``$PLAIN_CACHE`` when set, otherwise a scratch directory under
        ``/tmp``. Created if absent.

    Notes
    -----
    Pass the result as ``save_path`` to ``MultisliceCalculator.setup``.
    """
    root = Path(os.environ.get("PLAIN_CACHE", "/tmp/plain-simulation-cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root
