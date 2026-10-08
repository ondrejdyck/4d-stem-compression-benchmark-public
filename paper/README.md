# The manuscript

The paper, its Supporting Information, and every figure and table they include.

| | |
|---|---|
| `manuscript.pdf` | the manuscript |
| `supporting-information.pdf` | Sections S1 to S3: the simulated dataset's parameters, the code-length calculations, and the software |
| `generated/figures/` | `figure_1.pdf` to `figure_7.pdf`, as included |
| `generated/tables/` | the five generated tables, as LaTeX |
| `generated/tables_ascii/` | the same tables as plain text |
| `generated/tables_csv/` | the same tables as CSV |

Table 2 is written directly in the manuscript and has no generator, which is why there are five table files for six tables.

## These are outputs, not sources

Everything in `generated/` is written by the code in this repository and is never edited by hand. To rebuild it:

```bash
cd implementation/src
uv run python -m paper_artifacts.generate_all
```

Twelve artifacts in about ten seconds. `implementation/src/paper_artifacts/README.md` says which script produces which, and what each one reads.

Figure 6 and Table 6 rest on a simulated dataset that is not committed — it is several hundred megabytes and regenerable exactly. Without it those two are skipped rather than failing the run; `implementation/src/paper_artifacts/simulation/README.md` covers generating it.

## LaTeX sources are not published here

This repository carries the manuscript as PDF. The `.tex` sources, the bibliography and the tracked-changes build live in the authors' working repository and are not part of the public release.

The ASCII and CSV copies of each table exist so that the numbers can be read or parsed without going through either the PDF or LaTeX.
