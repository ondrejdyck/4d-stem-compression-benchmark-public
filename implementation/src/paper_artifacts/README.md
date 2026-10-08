# Paper artifacts

Every figure and table in the manuscript, and the code that produces it. Nothing here is drawn by hand or edited after generation.

## Regenerate everything

```bash
cd implementation/src
uv run python -m paper_artifacts.generate_all
```

Twelve artifacts, about ten seconds. Figure 6 and Table 6 need the simulated dataset; if it is absent they are skipped with an explanation rather than failing the run.

## What produces what

| manuscript | generator |
|---|---|
| Figure 1 — cross-dataset performance | `figures/combined_performance.py` |
| Figure 2 — multi-dimensional comparison | `figures/radar_chart.py` |
| Figure 3 — chunking strategy | `figures/chunking_comparison.py` |
| Figure 4 — sparsity against compression | `figures/sparsity_compression.py` |
| Figure 5 — modes of inference | `figures/panel_inference_modes.py` |
| Figure 6 — the simulated dataset | `figures/simulated_dataset.py` |
| Figure 7 — event-driven detection | `figures/event_detection.py` |
| Table 1 — datasets | `tables/tab_methods_datasets.py` |
| Table 3 — dataset characteristics | `tables/tab_dataset_summary.py` |
| Table 4 — implementation families | `tables/tab_implementation_families.py` |
| Table 5 — chunking strategy | `tables/tab_chunking_summary.py` |
| Table 6 — cost of storing the simulated cube | `tables/tab_generative_codelength.py` |

Table 2 is written directly in the manuscript and has no generator.

Every figure lands at `paper/generated/figures/figure_N.pdf`, written by its generator. There is no copy step, and no generator picks its own location — `outputs.py` holds the one function that writes them. Tables land in `paper/generated/tables{,_ascii,_csv}/`, and the section files `\input` the LaTeX.

The numbering is on the outputs and not on the scripts. A figure's number is a fact about the manuscript, so it belongs on the file the manuscript includes; a script's name is a fact about the code and should survive the paper reordering its figures.

## Running one at a time

Everything runs as a module from `implementation/src` — the root `pyproject.toml` sets `package = false`, so that directory has to be the working directory for `paper_artifacts` to be importable:

```bash
cd implementation/src
uv run python -m paper_artifacts.figures.combined_performance
uv run python -m paper_artifacts.tables.tab_dataset_summary
```

Every generator that reads benchmark output takes `--results-dir`, defaulting to the repository's `results/`. Those that read one particular file also take a flag naming it, which wins when both are given:

| | `--results-dir` | file override | reads |
|---|---|---|---|
| Figures 1 to 4 | yes | — | `aggregated/statistics.csv` |
| Table 1 | yes | `--dataset-inventory` | `dataset_inventory.csv` |
| Table 3 | yes | `--dataset-inventory`, `--statistics` | both |
| Tables 4, 5 | yes | `--statistics` | `aggregated/statistics.csv` |
| Table 6 | — | `--npz` | the simulated dataset |
| Figures 5, 7 | — | — | nothing |

When an input is missing, all eight say so and name the command that writes it, rather than raising from inside pandas. Figure 6 and Table 6 report the absent simulated dataset the same way.

PDF is the artifact. `--preview` also writes PNG and SVG, for a talk or for opening in a vector editor; neither is committed. Figures 5 to 7 have no argument parser, and read `FIGURE_PREVIEW=1` instead.

Figures 5 and 7 are numpy and matplotlib only. Figure 5 draws Peirce's three modes of inference; Figure 7 makes interpretive reduction concrete, the analog trace discarded and a claim about the event kept.

## The simulated dataset

Figure 6 and Table 6 rest on a simulated cube rather than measured data, and that is the point: its generating description is known exactly, so the shortest code for the counts can be computed rather than estimated. `simulation/README.md` covers the physics, the environment knobs, why the PySlice tag is pinned, and what the simulation cannot express.

`verify_reproduction.py` confirms a regenerated dataset is the published one, hashing the counts, the enlarged lambda patterns, the virtual images and the atom positions against `digests.json` and exiting non-zero on any difference. It needs numpy alone, which is why it sits here rather than inside `simulation/`.

## Fonts

Every figure script sets `pdf.fonttype = 42`. Matplotlib's default is Type 3, which several journals' production systems reject and which carries no ToUnicode map, so text in the figure cannot be selected, searched or read aloud. Check with `pdffonts` after adding a generator.
