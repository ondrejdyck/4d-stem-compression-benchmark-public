"""Simulate the 4D-STEM dataset shown in the manuscript's Discussion.

Deliberately unremarkable, so that one paragraph describes it with no caveats:
a **pristine** monolayer 2H-WSe2, **species-specific** Kirkland form factors
for real W and real Se, and **stock PySlice** multislice.

Emitted twice: the noiseless expected intensity ``lambda`` and one Poisson draw
from it. That pair is what the figure is about, and it is also what makes the
code-length comparison possible: knowing ``lambda`` exactly, the shortest code
for the counts can be computed rather than estimated. Both the expected length
and the length this particular draw costs are accumulated as the scan streams,
so the 1.6 GB of ``lambda`` never has to be held or stored.

Two passes. The first streams the whole scan to build the virtual-detector
images without ever holding the cube; the second re-simulates a handful of
probe positions chosen *from* those images, so the enlarged-pattern panels sit
on a tungsten column, a selenium column and a hollow site by measurement rather
than by guess.

Run (needs the ``simulation`` optional dependencies; see README.md)::

    FIGURE_DATA_DIR=/path/for/output uv run --extra simulation \\
        python -m paper_artifacts.simulation.simulate_dataset

A few minutes at 128x128 on an RTX A5000. The output npz is ~5 MB with the
counts cube included, which is ~99% zeros.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from paper_artifacts.simulation import config
from paper_artifacts.simulation.diffraction import simulate
from paper_artifacts.simulation.dose import electrons_per_probe, expected_counts, make_rng, sample_counts
from paper_artifacts.simulation.potential import grid_spec
from paper_artifacts.simulation.codelength import (
    poisson_entropy_bits, poisson_neglogp_bits)
from paper_artifacts.simulation.scan import scan_positions
from paper_artifacts.simulation.specimen import build_wse2_monolayer, thermal_trajectory

N_SCAN = int(os.environ.get("PLAIN_N_SCAN", "128"))
BATCH = int(os.environ.get("PLAIN_BATCH", "64"))
COUNT_SEED = int(os.environ.get("PLAIN_COUNT_SEED", "20260921"))

#: Save the full counts cube as well. It is ~99% zeros so it compresses to a
#: fraction of its 407 MB; the full lambda does not compress and is 1.63 GB,
#: so it is never saved in bulk -- only for the chosen positions.
SAVE_CUBE = os.environ.get("PLAIN_SAVE_CUBE", "1") != "0"

#: The counts cube is 407 MB uncompressed, so it is written outside the
#: repository by default. Override with ``FIGURE_DATA_DIR``.
OUT_DIR = Path(os.environ.get(
    "FIGURE_DATA_DIR", Path.home() / "4dstem-figure-data"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    trajectory, record = build_wse2_monolayer(min_lateral_A=config.MIN_LATERAL_A)
    spec = grid_spec(trajectory)
    scan = scan_positions(spec, N_SCAN, N_SCAN)
    n_pos = len(scan)
    shape = (N_SCAN, N_SCAN)

    atom_xyz = np.asarray(trajectory.positions)
    if atom_xyz.ndim == 3:
        atom_xyz = atom_xyz[0]
    species = np.array([str(t) for t in trajectory.atom_types])
    box = np.diag(np.asarray(trajectory.box_matrix)).copy()

    # One thermal state for the whole scan. Redrawing per batch would make
    # every probe position see a different specimen.
    displaced = thermal_trajectory(
        trajectory, n_configurations=config.FROZEN_PHONON_CONFIGS,
        sigma_A=config.FROZEN_PHONON_SIGMA_A, seed=config.FROZEN_PHONON_SEED)

    dose = electrons_per_probe()
    rng = make_rng(COUNT_SEED)

    def run(points):
        return simulate(
            displaced, spec, positions=[tuple(p) for p in points],
            n_configurations=config.FROZEN_PHONON_CONFIGS,
            sigma_A=config.FROZEN_PHONON_SIGMA_A,
            seed=config.FROZEN_PHONON_SEED, displace=False)

    theta = None
    bf_lam = np.zeros(n_pos); adf_lam = np.zeros(n_pos)
    bf_cnt = np.zeros(n_pos); adf_cnt = np.zeros(n_pos)
    # Code length of each frame under a coder that knows lambda exactly:
    # the expected length (entropy) and the length this draw actually costs.
    ideal_bits = np.zeros(n_pos); coded_bits = np.zeros(n_pos)
    cube = None
    nonzero = total_px = 0
    max_count = 0

    print(f"pass 1: streaming {n_pos} positions", flush=True)
    for start in range(0, n_pos, BATCH):
        diff = run(scan[start:start + BATCH])
        if theta is None:
            theta = diff.theta_rad
            bf_mask = (theta <= config.APERTURE_RAD).ravel()
            adf_mask = ((theta >= config.ADF_INNER_RAD)
                        & (theta <= config.ADF_OUTER_RAD)).ravel()
            nkx, nky = theta.shape
            print(f"  detector {nkx}x{nky} = {nkx * nky} pixels")
            if SAVE_CUBE:
                cube = np.empty((n_pos, nkx, nky), dtype=np.uint8)
        for offset in range(diff.intensity.shape[0]):
            k = start + offset
            lam = expected_counts(diff, n_electrons=dose, position=offset)
            counts = sample_counts(lam, rng)
            if counts.max() > 255:
                raise ValueError(f"count {counts.max()} exceeds uint8 at {k}")
            flat_l, flat_c = lam.ravel(), counts.ravel()
            bf_lam[k], adf_lam[k] = flat_l[bf_mask].sum(), flat_l[adf_mask].sum()
            bf_cnt[k], adf_cnt[k] = flat_c[bf_mask].sum(), flat_c[adf_mask].sum()
            ideal_bits[k] = poisson_entropy_bits(lam).sum()
            coded_bits[k] = poisson_neglogp_bits(lam, counts).sum()
            nonzero += int((counts > 0).sum()); total_px += counts.size
            max_count = max(max_count, int(counts.max()))
            if SAVE_CUBE:
                cube[k] = counts.astype(np.uint8)
        if start % (BATCH * 32) == 0:
            print(f"  {start + diff.intensity.shape[0]}/{n_pos}  "
                  f"{time.time() - started:.0f}s", flush=True)

    # --- choose the enlarged-pattern positions from the measured ADF -------
    adf_img = adf_lam.reshape(shape)
    order = np.argsort(adf_lam)
    picks = {
        "tungsten_column": int(order[-1]),
        "selenium_column": int(np.argmin(np.abs(
            adf_lam - np.percentile(adf_lam[adf_lam > np.median(adf_lam)], 35)))),
        "hollow_site": int(order[0]),
    }
    print(f"\npass 2: re-simulating {len(picks)} chosen positions {picks}")
    idx = list(picks.values())
    diff = run(scan[idx])
    pick_lam = np.stack([expected_counts(diff, n_electrons=dose, position=o)
                         for o in range(len(idx))]).astype(np.float32)
    pick_rng = make_rng(COUNT_SEED + 1)
    pick_counts = np.stack([sample_counts(l.astype(np.float64), pick_rng)
                            for l in pick_lam]).astype(np.uint8)

    step = (float(box[0] / N_SCAN), float(box[1] / N_SCAN))
    nyquist = config.WAVELENGTH_A / (4.0 * config.APERTURE_RAD)
    probe_d = 1.22 * config.WAVELENGTH_A / config.APERTURE_RAD
    metadata = {
        "description": "Simulated 4D-STEM of a pristine monolayer 2H-WSe2, "
                       "emitted as the noiseless expected intensity and as one "
                       "Poisson draw from it at the stated dose.",
        "specimen": "pristine 2H-WSe2 monolayer; no defects, no substitutions, "
                    "no weight overrides. Species-specific Kirkland form "
                    "factors for W and Se.",
        "simulator": "PySlice multislice (stock), frozen-phonon averaged in "
                     "intensity.",
        "voltage_V": config.VOLTAGE_V,
        "wavelength_A": config.WAVELENGTH_A,
        "convergence_semiangle_rad": config.APERTURE_RAD,
        "probe_diameter_A": probe_d,
        "probe_aberrations": "none; ideal in-focus probe.",
        "band_limit_rad": config.THETA_MAX_RAD,
        "adf_inner_rad": config.ADF_INNER_RAD,
        "adf_outer_rad": config.ADF_OUTER_RAD,
        "beam_current_A": config.BEAM_CURRENT_A,
        "dwell_s": config.DWELL_S,
        "dose_electrons_per_position": dose,
        "frozen_phonon_sigma_A": config.FROZEN_PHONON_SIGMA_A,
        "frozen_phonon_configs": config.FROZEN_PHONON_CONFIGS,
        "frozen_phonon_seed": config.FROZEN_PHONON_SEED,
        "frozen_phonon_convention": "sigma is the RMS displacement MAGNITUDE; "
                                    "each Cartesian component is drawn from "
                                    "N(0, sigma/sqrt(3)).",
        "count_seed": COUNT_SEED,
        "sampling_x_A": spec.sampling_x_A,
        "sampling_y_A": spec.sampling_y_A,
        "real_grid": [len(spec.xs), len(spec.ys)],
        "n_slices": spec.n_slices,
        "slice_thickness_A": spec.realised_slice_thickness_A,
        "scan_shape": list(shape),
        "scan_step_A": list(step),
        "scan_nyquist_step_A": nyquist,
        "scan_oversampling": [nyquist / step[0], nyquist / step[1]],
        "box_A": box.tolist(),
        "n_atoms": int(len(atom_xyz)),
        "lattice_a_A": 3.28, "se_se_thickness_A": 3.34,
        "vacuum_per_side_A": 2.0, "polytype": "2H",
        "build_record": record,
        "pattern_positions": picks,
        "nonzero_pixels": nonzero, "total_pixels": total_px,
        "max_count": max_count,
        "mean_counts_per_pixel": float(bf_cnt.sum() + adf_cnt.sum()) / total_px,
        "lambda_note": "scaled by the INCIDENT total, not by each pattern's own "
                       "sum, so lambda.sum() is position-dependent; the ~1.7% "
                       "scattered past the anti-aliasing aperture is kept.",
        "counts_note": "independent Poisson per pixel; the total is itself "
                       "Poisson, not fixed.",
        "cube_saved": bool(SAVE_CUBE),
        "ideal_code_bits": float(ideal_bits.sum()),
        "coded_bits_this_draw": float(coded_bits.sum()),
        "code_length_note": "ideal_code_bits is sum_i H(Poisson(lambda_i)) "
                            "over every detector pixel and probe position: the "
                            "expected length of the shortest code for the counts "
                            "given an exact model of the instrument and none of "
                            "the specimen. coded_bits_this_draw is the length "
                            "this particular Poisson realisation costs such a "
                            "coder. Neither is achievable in an experiment, "
                            "where lambda is unknown; both are lower bounds.",
    }

    arrays = dict(
        adf_from_lambda=adf_lam.reshape(shape), adf_from_counts=adf_cnt.reshape(shape),
        bf_from_lambda=bf_lam.reshape(shape), bf_from_counts=bf_cnt.reshape(shape),
        pattern_lambda=pick_lam, pattern_counts=pick_counts,
        pattern_index=np.array(idx), pattern_labels=np.array(list(picks)),
        ideal_code_bits_per_position=ideal_bits.reshape(shape),
        coded_bits_per_position=coded_bits.reshape(shape),
        theta_rad=theta.astype(np.float32),
        theta_x_rad=diff.theta_x_rad.astype(np.float32),
        theta_y_rad=diff.theta_y_rad.astype(np.float32),
        scan_positions=scan.astype(np.float64), scan_shape=np.array(shape),
        atom_positions=atom_xyz.astype(np.float64), atom_species=species,
        box_A=box, metadata_json=np.array(json.dumps(metadata, indent=2)))
    if SAVE_CUBE:
        arrays["counts"] = cube

    out = OUT_DIR / f"wse2_pristine_{N_SCAN}x{N_SCAN}_{dose:.0f}e.npz"
    np.savez_compressed(out, **arrays)

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"nonzero {nonzero:,} of {total_px:,} ({100 * nonzero / total_px:.2f}%), "
          f"max count {max_count}")
    ib, cb = ideal_bits.sum(), coded_bits.sum()
    print(f"ideal code length {ib / 8 / 1e6:.3f} MB (expected), "
          f"{cb / 8 / 1e6:.3f} MB (this draw), "
          f"agree to {abs(cb - ib) / ib:.2e}); "
          f"raw cube {total_px / 1e6:.1f} MB -> {total_px * 8 / ib:.1f}x")
    print(f"scan step {step[0]:.4f} x {step[1]:.4f} A; Nyquist {nyquist:.4f} A; "
          f"oversampled {nyquist / step[0]:.2f}x / {nyquist / step[1]:.2f}x")
    print(f"probe diameter {probe_d:.3f} A -> adjacent probes overlap "
          f"{100 * (1 - step[0] / probe_d):.1f}% along x")
    print(f"elapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
