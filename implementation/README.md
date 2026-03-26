# Implementation

This directory contains the code used to generate the paper artifacts and to run the benchmark on local 4D-STEM datasets.

## Contents

- `src/compression_benchmark.py` — core benchmark engine
- `src/run_benchmark.py` — single-dataset CLI
- `src/run_all_benchmarks.py` — batch runner over local `.emd` files
- `src/run_multiple_benchmarks.py` — repeated runs for variability analysis
- `src/aggregate_multi_run_results.py` — combines repeated-run outputs
- `src/data_loader.py` — shared result-loading utilities
- `src/Fig*.py` — figure generation scripts
- `src/paper_artifacts/` — table and inventory generation scripts
- `src/smoke_test_public.py` — verifies the trimmed public workflow using the included fixture

## Data

The public repo does not include raw benchmark datasets. The benchmark scripts expect local `.emd` files when run against real data.

For the original study, datasets were stored under `implementation/data/`.

The committed `results/` CSV files are sufficient to regenerate the paper tables and figures.

## Typical usage

Run a single dataset:

```bash
python src/run_benchmark.py /path/to/dataset.emd
```

Run all local datasets:

```bash
python src/run_all_benchmarks.py --data-dir /path/to/data --yes
```

Generate paper tables/figures from existing results:

These commands are deterministic and operate only on the committed CSV outputs.

```bash
python src/paper_artifacts/datasets/build_dataset_inventory.py --data-dir /path/to/data
python src/paper_artifacts/tables/Tab1_datasets.py
python src/paper_artifacts/tables/Tab3_dataset_summary.py
python src/paper_artifacts/tables/Tab4_implementation_families.py
python src/paper_artifacts/tables/Tab5_chunking_summary.py
python src/Fig2_radar_chart.py
python src/Fig1_combined_performance.py
python src/Fig4_chunking_comparison.py
python src/Fig3_sparsity_vs_compression.py
python src/smoke_test_public.py
```
