# Compression benchmark

Measures compression ratio and read/write throughput for thirteen lossless implementations against 4D-STEM datasets, under three chunking strategies. This is what produces the numbers every figure and table in the manuscript is built from.

Table 2 of the manuscript is the authoritative list of implementations; Table 1 describes the five datasets used.

## Data in

Datasets go in `data/`, as EMD 1.0 or any HDF5 file with a 4D array. EMD 1.0 puts the cube at `/version_1/data/datacubes/datacube_000/data`; other layouts are found by shape.

**The datasets behind the manuscript are not included.** They run from 8 MiB to 8 GiB and are not ours to publish. What is committed in `results/` is the aggregated output of the ten-run sweep over them, which is all the figures and tables need — so every artifact in `paper/generated/` rebuilds from this repository alone.

To exercise the benchmark itself without them, `fixtures/smoke_test.emd` is a small synthetic file that runs the same paths:

```bash
uv run python src/smoke_test_public.py
```

`compression_benchmark.py` also falls back to that fixture when no dataset is given, so the repository runs end to end out of the box.

## One run

```bash
uv run python src/run_benchmark.py data/<dataset>.emd
uv run python src/run_benchmark.py data/<dataset>.emd --name <label> --output <dir>
```

Writes `results/<name>/benchmark_results.csv`, `metadata.json`, and a readable summary. `--help` lists the rest.

## Ten runs, which is what the paper reports

Every error bar in the manuscript is a min–max range over ten independent runs of the same implementation on the same dataset. Compression ratio is deterministic and does not vary; the timings do, and the paper reports their spread rather than a single number.

```bash
uv run python src/run_multiple_benchmarks.py --n-runs 10
uv run python src/aggregate_multi_run_results.py
```

The first loops; the second collapses the runs into `results/aggregated/statistics.csv`, which is the one file the figures and tables read. Both take `--help`.

A full ten-run sweep over all five datasets is hours of work, most of it in gzip-9. `run_multiple_benchmarks.py --start-run N` resumes an interrupted sweep rather than starting over.

## What comes out

```
results/
├── <dataset>/                  a single run writes here
│   ├── benchmark_results.csv
│   ├── metadata.json
│   └── <dataset>_detailed_results.txt
├── run_NNN_<timestamp>/        the sweep writes one of these per run,
│   └── <dataset>/              each holding the same per-dataset directories
├── aggregated/
│   ├── statistics.csv          mean, sd, min, max, median, CV% per method
│   ├── all_runs_combined.csv
│   └── summary_report.txt
└── dataset_inventory.csv       shape, dtype, sparsity, max value per dataset
```

`metadata.json` is per-run and is not published; `statistics.csv` and `dataset_inventory.csv` are. Anything that needs sparsity reads it from the inventory for that reason.

## Code

| file | role |
|---|---|
| `src/compression_benchmark.py` | the measurement itself — writes, reads, times |
| `src/run_benchmark.py` | one dataset, one run |
| `src/run_all_benchmarks.py` | every dataset in `data/`, one run each |
| `src/run_multiple_benchmarks.py` | the ten-run sweep |
| `src/aggregate_multi_run_results.py` | runs → `aggregated/statistics.csv` |
| `src/paper_artifacts/datasets/build_dataset_inventory.py` | writes `dataset_inventory.csv` |

The figure and table generators are documented separately, in `src/paper_artifacts/README.md`.
