#!/usr/bin/env python3
"""Generate Table 3 (chunking summary) in LaTeX/ASCII/CSV.

This script produces the chunking strategy summary table (manuscript label
\\label{tab:chunking_summary}) in three synchronized formats:

- LaTeX:  paper/generated/tables/table_chunking_summary.tex
- ASCII:  paper/generated/tables_ascii/table_chunking_summary.txt
- CSV:    paper/generated/tables_csv/table_chunking_summary.csv

Inputs (source of truth):
- results/aggregated/statistics.csv

Method
------
For the 4D_Diff dataset, we group all chunked HDF5 configurations by chunking
prefix (real_space, balanced, single_frame) and compute the mean of:

- compression_ratio_mean
- write_throughput_gbs_mean
- read_throughput_gbs_mean

across the 13 compression implementations available for each chunking strategy.

Sparsity is defined by exact zeros; all benchmarks are lossless.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Row:
    strategy: str
    chunk_size: str
    mean_ratio: float
    mean_write_gbs: float
    mean_read_gbs: float
    n_methods: int


def _repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def compute_rows(stats: pd.DataFrame, dataset: str = "4D_Diff") -> list[Row]:
    chunking_defs = [
        ("real_space", "Real-space", "(32, 32, 256, 256)"),
        ("balanced", "Balanced", "(16, 16, 128, 128)"),
        ("single_frame", "Single-frame", "(1, 1, 256, 256)"),
    ]

    rows: list[Row] = []
    for prefix, display, chunk in chunking_defs:
        sub = stats[
            (stats["dataset"] == dataset)
            & (stats["method"].str.startswith(prefix + "_"))
        ]
        if sub.empty:
            raise ValueError(f"No rows for dataset={dataset}, chunking={prefix}")

        rows.append(
            Row(
                strategy=display,
                chunk_size=chunk,
                mean_ratio=float(sub["compression_ratio_mean"].mean()),
                mean_write_gbs=float(sub["write_throughput_gbs_mean"].mean()),
                mean_read_gbs=float(sub["read_throughput_gbs_mean"].mean()),
                n_methods=int(sub["method"].nunique()),
            )
        )

    return rows


def write_csv_out(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "strategy",
                "chunk_size",
                "mean_ratio",
                "mean_write_gbs",
                "mean_read_gbs",
                "n_methods",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.strategy,
                    r.chunk_size,
                    f"{r.mean_ratio:.6f}",
                    f"{r.mean_write_gbs:.6f}",
                    f"{r.mean_read_gbs:.6f}",
                    r.n_methods,
                ]
            )


def write_ascii(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Strategy",
        "Chunk Size",
        "Mean Ratio",
        "Mean Write (GiB/s)",
        "Mean Read (GiB/s)",
        "n",
    ]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r.strategy,
                r.chunk_size,
                f"{r.mean_ratio:.2f}x",
                f"{r.mean_write_gbs:.2f}",
                f"{r.mean_read_gbs:.2f}",
                str(r.n_methods),
            ]
        )

    widths = [
        max(len(h), max(len(row[i]) for row in table_rows))
        for i, h in enumerate(headers)
    ]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    lines = [fmt_row(headers), sep]
    for row in table_rows:
        lines.append(fmt_row(row))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Chunking strategy impact on performance (8 GiB 4D Diff. dataset, averaged across all implementations)}"
    )
    lines.append("\\label{tab:chunking_summary}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\hline")
    lines.append(
        "Strategy & Chunk Size & Mean Ratio & Mean Write (GiB/s) & Mean Read (GiB/s) \\\\"
    )
    lines.append("\\hline")

    for r in rows:
        ratio = f"{r.mean_ratio:.2f}$\\times$"
        lines.append(
            f"{r.strategy} & {r.chunk_size} & {ratio} & {r.mean_write_gbs:.2f} & {r.mean_read_gbs:.2f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chunking summary table")
    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
        help="Path to aggregated statistics.csv (default: results/aggregated/statistics.csv)",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_script(Path(__file__))
    stats_path = args.statistics or (
        repo_root / "results" / "aggregated" / "statistics.csv"
    )

    stats = pd.read_csv(stats_path)
    rows = compute_rows(stats)

    out_tex = (
        repo_root / "paper" / "generated" / "tables" / "Table5_chunking_summary.tex"
    )
    out_txt = (
        repo_root
        / "paper"
        / "generated"
        / "tables_ascii"
        / "Table5_chunking_summary.txt"
    )
    out_csv = (
        repo_root / "paper" / "generated" / "tables_csv" / "Table5_chunking_summary.csv"
    )

    write_latex(rows, out_tex)
    write_ascii(rows, out_txt)
    write_csv_out(rows, out_csv)

    print("✓ Generated Table 5 outputs:")
    print(f"  LaTeX: {out_tex}")
    print(f"  ASCII: {out_txt}")
    print(f"  CSV:   {out_csv}")


if __name__ == "__main__":
    main()
