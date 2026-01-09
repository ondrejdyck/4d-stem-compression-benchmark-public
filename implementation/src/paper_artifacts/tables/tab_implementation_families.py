#!/usr/bin/env python3
"""Generate Table 4 (implementation family comparison) in LaTeX/ASCII/CSV.

This script reproduces the manuscript table labeled:
  \\label{tab:algorithm_families}

Outputs (synchronized):
- LaTeX:  paper/generated/tables/Table4_implementation_families.tex
- ASCII:  paper/generated/tables_ascii/Table4_implementation_families.txt
- CSV:    paper/generated/tables_csv/Table4_implementation_families.csv

Input (source of truth):
- results/aggregated/statistics.csv

Method
------
For each implementation family, we use the balanced-chunking configuration and
aggregate across the five datasets:
- Mean Ratio = mean(compression_ratio_mean)
- Range = min--max of compression_ratio_mean
- Mean Write (s) = mean(write_time_mean)

All values are computed directly from the aggregated 10-run statistics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Row:
    family: str
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    mean_write_s: float


def _repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def compute_rows(stats: pd.DataFrame) -> list[Row]:
    # Map display label -> balanced method name
    families = [
        ("Blosc (zlib)", "balanced_blosc_zlib"),
        ("Blosc (zstd)", "balanced_blosc_zstd"),
        ("Gzip-9", "balanced_gzip_9"),
        ("Gzip-6", "balanced_gzip_6"),
        ("Blosc (lz4hc)", "balanced_blosc_lz4hc"),
        ("Szip", "balanced_szip"),
        ("Blosc (lz4)", "balanced_blosc_lz4"),
    ]

    rows: list[Row] = []
    for label, method in families:
        sub = stats[stats["method"] == method]
        if sub.empty:
            raise ValueError(f"No rows found for method={method}")

        ratios = sub["compression_ratio_mean"].astype(float)
        rows.append(
            Row(
                family=label,
                mean_ratio=float(ratios.mean()),
                min_ratio=float(ratios.min()),
                max_ratio=float(ratios.max()),
                mean_write_s=float(sub["write_time_mean"].astype(float).mean()),
            )
        )

    return rows


def write_csv_out(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "mean_ratio", "min_ratio", "max_ratio", "mean_write_s"])
        for r in rows:
            w.writerow(
                [
                    r.family,
                    f"{r.mean_ratio:.6f}",
                    f"{r.min_ratio:.6f}",
                    f"{r.max_ratio:.6f}",
                    f"{r.mean_write_s:.6f}",
                ]
            )


def write_ascii(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Implementation Family", "Mean Ratio", "Range", "Mean Write (s)"]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r.family,
                f"{r.mean_ratio:.1f}x",
                f"{r.min_ratio:.1f}--{r.max_ratio:.1f}x",
                f"{r.mean_write_s:.1f}",
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
        "\\caption{Implementation family performance comparison (balanced chunking, averaged across all 5 datasets)}"
    )
    lines.append("\\label{tab:algorithm_families}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\hline")
    lines.append("Implementation Family & Mean Ratio & Range & Mean Write (s) \\\\")
    lines.append("\\hline")

    for r in rows:
        mean_ratio = f"{r.mean_ratio:.1f}$\\times$"
        range_ratio = f"{r.min_ratio:.1f}--{r.max_ratio:.1f}$\\times$"
        lines.append(
            f"{r.family} & {mean_ratio} & {range_ratio} & {r.mean_write_s:.1f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate implementation family table")
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
        repo_root
        / "paper"
        / "generated"
        / "tables"
        / "Table4_implementation_families.tex"
    )
    out_txt = (
        repo_root
        / "paper"
        / "generated"
        / "tables_ascii"
        / "Table4_implementation_families.txt"
    )
    out_csv = (
        repo_root
        / "paper"
        / "generated"
        / "tables_csv"
        / "Table4_implementation_families.csv"
    )

    write_latex(rows, out_tex)
    write_ascii(rows, out_txt)
    write_csv_out(rows, out_csv)

    print("✓ Generated Table 4 outputs:")
    print(f"  LaTeX: {out_tex}")
    print(f"  ASCII: {out_txt}")
    print(f"  CSV:   {out_csv}")


if __name__ == "__main__":
    main()
