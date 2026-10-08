# Simulated 4D-STEM dataset

Everything needed to regenerate the dataset shown in Figure 6 of the
manuscript and the code-length comparison in Table 6 beside it.

## What this is

A multislice simulation of a pristine monolayer 2H-WSe<sub>2</sub>, scanned at
128×128 probe positions onto a 147×169 detector at 374.5 electrons per
position. The specimen is built from crystallography alone — no planted
defects, no substitutions, no weight overrides — with species-specific Kirkland
form factors for real W and real Se.

The simulation emits the noiseless expected intensity λ **and** one Poisson
draw from it. That pairing is what makes the manuscript's comparison possible:
knowing λ exactly, the shortest code for the counts can be *computed* rather
than estimated. Both the expected code length Σ H(Poisson(λ)) and the length
this particular draw actually costs are accumulated as the scan streams, so the
1.6 GB of λ never has to be held in memory or written to disk.

## Running it

```bash
cd implementation/src
export FIGURE_DATA_DIR=/path/for/output
uv sync --extra simulation
uv run python -m paper_artifacts.simulation.simulate_dataset
```

The `cd` matters: the root `pyproject.toml` sets `package = false`, so
`paper_artifacts` is importable only when its parent is on the path.

A few minutes at 128×128 on an RTX A5000. Output is a ~5 MB `.npz`; the
407 MB counts cube is inside it and is 98.9% zeros. Then:

```bash
uv run python -m paper_artifacts.tables.tab_generative_codelength
```

To confirm a regenerated dataset is the published one:

```bash
uv run python -m paper_artifacts.verify_reproduction
```

It hashes the counts, the enlarged lambda patterns, the virtual images and the
atom positions against digests committed in `paper_artifacts/digests.json`, and
exits non-zero if any differ. It needs only numpy, not PySlice, which is why it
sits outside this subpackage.

Knobs, all via environment: `PLAIN_N_SCAN` (default 128), `PLAIN_BATCH` (64),
`PLAIN_COUNT_SEED` (20260921), `PLAIN_SAVE_CUBE` (set to `0` to skip the cube
and keep only the virtual images and the code lengths), `PLAIN_PRECISION`
(`single`), and `PLAIN_CACHE` for PySlice's wavefunction scratch directory.

## The physics

The multislice physics is [PySlice](https://github.com/sea-ecosystem/PySlice)'s, pinned
at tag `4dstem-compression-2026`. This package supplies the specimen, the grid,
the dose and Poisson sampling, the scan loop, and the code-length accumulation.

Importing it selects single precision. PySlice defaults to float64 on every
device but MPS, which costs 3.8x end to end on Ampere for a 1.5e-7 relative
change in the ADF fraction. `_precision` records the measurements and sets
PySlice's own `PYSLICE_PRECISION` variable.

The pinned tag is required, not merely recorded. PySlice before it derived a
slice's upper bound and the next slice's lower bound from two different
floating-point expressions, so an atom lying within one ULP of a boundary was
silently dropped or counted twice. Atom planes sit at exact coordinates and
slice boundaries land on round numbers, so this is what a symmetric slab does by
construction rather than an edge case. Running against an earlier PySlice will
produce a different cube, and `verify_reproduction` will say so.

## Two things the simulation cannot express

Stated because their absence is silent, and a plausible-looking diffraction
pattern comes out either way:

- **No aberrations.** The probe here is ideal and exactly in focus. That is a
  choice, not a limitation: PySlice applies defocus and Cnm aberrations as
  methods on `Probe` (`defocus`, `aberrate`) rather than as constructor
  arguments, and neither is called here.
- **Detectors near the band limit drift with sampling.** The anti-aliasing
  cutoff is a soft roll-off applied at every slice, so any detector integral in
  the outer half of the representable range moves as the real-space sampling
  changes. The ADF detector used here (75–130 mrad against a 180 mrad band
  limit) sits at 0.72 of the cutoff. Representability margin is not convergence
  margin.
