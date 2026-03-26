# 4D STEM Compression Benchmark

This repository contains the public-facing materials for the 4D-STEM compression benchmarking project.
It includes the released manuscript PDF, generated paper artifacts, and the scripts used to reproduce key tables and figures.

Raw benchmark datasets are not included.

## What’s here

- `paper/` — final manuscript PDF and generated publication artifacts
- `implementation/` — benchmark and figure-generation code
- `results/` — aggregated benchmark outputs used by the manuscript
- `pyproject.toml` / `uv.lock` — reproducible Python environment

## Key outputs

- `paper/manuscript.pdf` — current public manuscript
- `paper/generated/` — canonical tables and figures used in the paper
- `results/aggregated/` — combined benchmark statistics
- `results/dataset_inventory.csv` — dataset summary used in the manuscript

## Reproducing artifacts

The code under `implementation/` can regenerate the benchmark summaries, tables, and figures from the packaged results.
See `implementation/README.md` for implementation-specific usage notes.

For a no-data smoke test, run `python implementation/src/smoke_test_public.py`.
The published figures and tables are generated deterministically from committed CSV outputs.

## arXiv submission bundle

Run `./prepare_arxiv_bundle.sh` to create a self-contained `arxiv-src/` staging directory and `arxiv-src.tar.gz` archive for submission. The staging directory is local-only and ignored by git.

Typical setup:

```bash
uv sync
```

## Repository layout

```text
.
├── implementation/
├── paper/
├── results/
├── pyproject.toml
└── uv.lock
```

## Citation

If you use this work, please cite the manuscript in `paper/manuscript.pdf`.

## Contact

Ondrej Dyck — dyckoe@ornl.gov
