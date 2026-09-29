# 4D STEM Compression Benchmark

Public materials for the 4D-STEM compression benchmarking project: the manuscript, the generated tables and figures, and the code that produces them.

## What's here

- `paper/` — the manuscript and every generated table and figure
- `implementation/` — the benchmark, the figure and table generators, and the multislice simulation
- `results/` — aggregated benchmark statistics, the inputs to the tables and figures
- `pyproject.toml`, `uv.lock`, `.python-version` — a pinned Python environment

## Which script makes which artifact

| Manuscript | Script |
|---|---|
| Table 1 — datasets | `implementation/src/paper_artifacts/tables/tab_methods_datasets.py` |
| Table 2 — compression implementations | not generated; written directly in the manuscript |
| Table 3 — dataset characteristics | `implementation/src/paper_artifacts/tables/tab_dataset_summary.py` |
| Table 4 — implementation families | `implementation/src/paper_artifacts/tables/tab_implementation_families.py` |
| Table 5 — chunking strategy | `implementation/src/paper_artifacts/tables/tab_chunking_summary.py` |
| Table 6 — cost of storing the simulated cube | `implementation/src/paper_artifacts/tables/tab_generative_codelength.py` |
| Figure 1 — cross-dataset performance | `implementation/src/plot_combined_performance.py` |
| Figure 2 — multi-dimensional comparison | `implementation/src/plot_radar_chart.py` |
| Figure 3 — chunking strategy | `implementation/src/plot_chunking_comparison.py` |
| Figure 4 — sparsity against compression | `implementation/src/plot_sparsity_compression.py` |
| Figure 5 — modes of inference | `implementation/src/paper_artifacts/figures/panel_inference_modes.py` |
| Figure 6 — the simulated dataset | `implementation/src/paper_artifacts/figures/simulated_dataset.py` |
| Figure 7 — event-driven detection | `implementation/src/paper_artifacts/figures/event_detection.py` |

Generated artifacts land in `paper/generated/`: `tables/` as LaTeX, `tables_ascii/` as plain text, `tables_csv/` as CSV, and `figures/` as vector PDF.

## Reproducing the tables and figures

Tables 1 and 3 to 5 and Figures 1 to 4 come from the committed statistics in `results/` and need no data beyond what is in this repository:

```bash
uv sync
uv run python implementation/src/paper_artifacts/tables/tab_dataset_summary.py
uv run python implementation/src/plot_combined_performance.py
```

and so on for the rest, following the table above.

## Reproducing the simulated dataset

Figure 6 and Table 6 rest on a simulated 4D-STEM dataset rather than on measured data, which is what makes the code-length comparison possible: the description that generated the cube is known exactly, so the shortest code for the counts can be computed rather than estimated.

```bash
cd implementation/src
export FIGURE_DATA_DIR=/path/for/output

uv sync --extra simulation
uv run python -m paper_artifacts.simulation.simulate_dataset
uv run python -m paper_artifacts.tables.tab_generative_codelength
uv run python -m paper_artifacts.verify_reproduction
```

The first writes the dataset, the second regenerates Table 6 from it, and the third confirms the result is the published one. `verify_reproduction` hashes the counts, the enlarged lambda patterns, the virtual images and the atom positions against digests committed here, and exits non-zero on any difference.

Only the first step needs the `simulation` extra, which requires Python 3.12 and a CUDA device. The other two need NumPy alone. The multislice physics is [PySlice](https://github.com/sea-ecosystem/PySlice), pinned by tag in the lock file.

## Data availability

**The raw 4D-STEM datasets are not included.** They are large, and the benchmark is reported from aggregated statistics rather than from the measurements themselves. What is committed in `results/` is sufficient to regenerate every table and figure in the manuscript that derives from measured data.

The benchmark scripts accept local `.emd` files if you have your own. To exercise the code without any data, `implementation/fixtures/smoke_test.emd` is a small synthetic file that runs the same paths:

```bash
uv run python implementation/src/smoke_test_public.py
```

The simulated dataset behind Figure 6 is not committed either. It is regenerable exactly, as above.

## arXiv submission bundle

`./prepare_arxiv_bundle.sh` stages a self-contained `arxiv-src/` directory and `arxiv-src.tar.gz`. Both are local-only and ignored by git.

## Citation

Please cite the manuscript in `paper/`.

## Contact

Ondrej Dyck — dyckoe@ornl.gov
