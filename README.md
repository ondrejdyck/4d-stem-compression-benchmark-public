# 4D STEM Compression Benchmark

A benchmark of thirteen lossless compression implementations on 4D-STEM datasets, and the manuscript built from it.

4D-STEM detectors produce data faster than it can be stored, moved or looked at. This measures how far lossless compression closes that gap — and finds that it does not, because compression is bounded by the entropy of the source and the best implementations are already close to that bound. What follows from this is a question about which data to keep, which the manuscript takes up.

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/). In a clone of this repository:

```bash
uv sync                       # benchmark, figures, tables
uv sync --extra simulation    # adds the multislice stack
```

Python 3.12 or later throughout; `.python-version` pins it. The `simulation` extra additionally needs a CUDA device.

## Where things are

Each part of the repository has its own README, covering how to reproduce what it produces.

| | |
|---|---|
| [`implementation/`](implementation/README.md) | the compression benchmark — how to run it, and what it writes |
| [`implementation/src/paper_artifacts/`](implementation/src/paper_artifacts/README.md) | every figure and table in the manuscript, and the code that produces it |
| [`implementation/src/paper_artifacts/simulation/`](implementation/src/paper_artifacts/simulation/README.md) | the simulated dataset behind Figure 6 and Table 6, and the physics behind it |
| [`paper/`](paper/README.md) | the manuscript and the generated artifacts it includes |

`results/` holds the benchmark output. `results/aggregated/statistics.csv` is what the benchmark-derived figures and tables are built from.

## Regenerate everything

```bash
cd implementation/src
uv run python -m paper_artifacts.generate_all
```

Twelve artifacts — seven figures and five tables — in about ten seconds. Figure 6 and Table 6 rest on the simulated dataset, which is not committed; if it is absent they are skipped with an explanation rather than failing the run.

## Citation

Manuscript under review. `CITATION.cff` carries the authors and title; each
release is archived on Zenodo with a DOI.

## Licence

MIT, see `LICENSE`. Authored by UT-Battelle, LLC under Contract
No. DE-AC05-00OR22725 with the U.S. Department of Energy.
