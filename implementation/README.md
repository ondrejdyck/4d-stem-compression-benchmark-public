# Implementation

This directory contains the code used to generate the paper artifacts and to run the benchmark on local 4D-STEM datasets.

## Contents

- `src/compression_benchmark.py` — core benchmark engine
- `src/run_benchmark.py` — single-dataset CLI
- `src/run_all_benchmarks.py` — batch runner over local `.emd` files
- `src/run_multiple_benchmarks.py` — repeated runs for variability analysis
- `src/aggregate_multi_run_results.py` — combines repeated-run outputs
- `src/data_loader.py` — shared result-loading utilities
- `src/plot_*.py` — the four benchmark figures
- `src/paper_artifacts/tables/` — the six generated tables
- `src/paper_artifacts/figures/` — the three conceptual figures
- `src/paper_artifacts/simulation/` — multislice simulation of the dataset behind Figure 6
- `src/paper_artifacts/verify_reproduction.py` — checks a regenerated dataset against committed digests
- `src/paper_artifacts/datasets/` — dataset inventory
- `src/smoke_test_public.py` — verifies the trimmed public workflow using the included fixture

## Data

The public repo does not include raw benchmark datasets. The benchmark scripts expect local `.emd` files when run against real data.

For the original study, raw datasets were stored locally and are not included here.

The committed `results/` CSV files are sufficient to regenerate the paper tables and figures.

## Typical usage

Run a single dataset:

```bash
uv run python src/run_benchmark.py /path/to/dataset.emd
```

Run all local datasets:

```bash
uv run python src/run_all_benchmarks.py --data-dir /path/to/data --yes
```

Generate paper tables/figures from existing results:

These commands are deterministic when run against the committed CSV outputs.

```bash
uv run python src/paper_artifacts/datasets/build_dataset_inventory.py --data-dir /path/to/data
uv run python src/paper_artifacts/tables/tab_methods_datasets.py
uv run python src/paper_artifacts/tables/tab_dataset_summary.py
uv run python src/paper_artifacts/tables/tab_implementation_families.py
uv run python src/paper_artifacts/tables/tab_chunking_summary.py
uv run python src/plot_combined_performance.py
uv run python src/plot_radar_chart.py
uv run python src/plot_chunking_comparison.py
uv run python src/plot_sparsity_compression.py
```

Smoke test the public workflow without raw data:

```bash
uv run python src/smoke_test_public.py
```

## The simulated dataset

Figure 6 and Table 6 rest on a simulated dataset rather than measured data. Regenerating it needs the `simulation` extra, Python 3.12 and a CUDA device:

```bash
cd src
export FIGURE_DATA_DIR=/path/for/output

uv sync --extra simulation
uv run python -m paper_artifacts.simulation.simulate_dataset
uv run python -m paper_artifacts.tables.tab_generative_codelength
uv run python -m paper_artifacts.verify_reproduction
```

`verify_reproduction` needs NumPy alone, not the simulation stack, so a dataset can be checked without a GPU. It exits non-zero if any array differs from the committed digests.

The three conceptual figures are generated from `src/paper_artifacts/figures/`. Figure 6's generator reads the dataset written above; the other two need no inputs.
