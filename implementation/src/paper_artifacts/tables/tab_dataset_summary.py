#!/usr/bin/env python3
"""Generate the dataset summary table in LaTeX/ASCII/CSV.

This script updates the public-release "version-of-record" artifacts:

- LaTeX:  paper/generated/tables/Table1_dataset_summary.tex
- ASCII:  paper/generated/tables_ascii/Table1_dataset_summary.txt
- CSV:    paper/generated/tables_csv/Table1_dataset_summary.csv

Inputs (source of truth):
- results/dataset_inventory.csv
- results/aggregated/statistics.csv

Notes
-----
- Dataset sizes are reported in binary GiB (1 GiB = 1024^3 bytes) using
  `uncompressed_bytes` from dataset_inventory.csv.
- Sparsity is reported as fraction of values exactly equal to zero.
- Best compression is selected by maximum compression_ratio_mean per dataset.

Manuscript mapping
------------------
This script reproduces the manuscript table labeled:
  \\label{tab:dataset_summary}
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Row:
    dataset: str
    size_gib: float
    sparsity_pct: float
    best_ratio: float
    implementation_display: str
    file_size_mib: float


def _repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _chunk_display(chunk: str) -> str:
    mapping = {
        "balanced": "balanced",
        "real_space": "real-space",
        "single_frame": "single-frame",
    }
    return mapping.get(chunk, chunk)


def _format_impl(method: str) -> str:
    if method.startswith("balanced_"):
        impl = method[len("balanced_") :]
        return _latex_escape(impl)

    if method.startswith("real_space_"):
        chunk = "real_space"
        impl = method[len("real_space_") :]
        return f"{_latex_escape(impl)} ({_chunk_display(chunk)})"

    if method.startswith("single_frame_"):
        chunk = "single_frame"
        impl = method[len("single_frame_") :]
        return f"{_latex_escape(impl)} ({_chunk_display(chunk)})"

    return _latex_escape(method)


def _format_size_for_table(size_gib: float) -> str:
    if size_gib < 0.1:
        return f"{size_gib:.3f}"
    return f"{size_gib:.1f}"


def build_rows(dataset_inventory: pd.DataFrame, stats: pd.DataFrame) -> list[Row]:
    inv = dataset_inventory.set_index("dataset_id")

    preferred_order = [
        "4D_EELS",
        "4D_Diff",
        "4D_Diff-2x2-binning",
        "4D_Diff-4x4-binning",
        "3D_EELS",
    ]
    datasets = [d for d in preferred_order if d in inv.index] + [
        d for d in sorted(inv.index) if d not in preferred_order
    ]

    rows: list[Row] = []
    for dataset in datasets:
        uncompressed_bytes = int(inv.loc[dataset, "uncompressed_bytes"])
        size_gib = uncompressed_bytes / (1024**3)
        sparsity_pct = float(inv.loc[dataset, "sparsity_fraction"]) * 100.0

        sub = stats[stats["dataset"] == dataset]
        if sub.empty:
            raise ValueError(f"No statistics found for dataset={dataset}")
        best = sub.loc[sub["compression_ratio_mean"].idxmax()]

        rows.append(
            Row(
                dataset=dataset,
                size_gib=size_gib,
                sparsity_pct=sparsity_pct,
                best_ratio=float(best["compression_ratio_mean"]),
                implementation_display=_format_impl(str(best["method"])),
                file_size_mib=float(best["file_size_mb_mean"]),
            )
        )

    return rows


def write_latex(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Dataset characteristics and best compression performance achieved. Chunking strategy is specified in parentheses where it differs from balanced.}"
    )
    lines.append("\\label{tab:dataset_summary}")
    lines.append("\\begin{tabular}{lrrrll}")
    lines.append("\\hline")
    lines.append(
        "Dataset & Size (GiB) & Sparsity (\\%) & Best Ratio & Implementation & File Size (MiB) \\\\"
    )
    lines.append("\\hline")

    for r in rows:
        dataset = _latex_escape(r.dataset)
        size = _format_size_for_table(r.size_gib)
        sparsity = f"{r.sparsity_pct:.1f}"
        ratio = f"{r.best_ratio:.1f}$\\times$"
        impl = r.implementation_display
        file_mib = f"{r.file_size_mib:.1f}"
        lines.append(
            f"{dataset} & {size} & {sparsity} & {ratio} & {impl} & {file_mib} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_ascii(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Dataset",
        "Size (GiB)",
        "Sparsity (%)",
        "Best Ratio",
        "Implementation",
        "File Size (MiB)",
    ]

    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r.dataset,
                _format_size_for_table(r.size_gib),
                f"{r.sparsity_pct:.1f}",
                f"{r.best_ratio:.1f}x",
                r.implementation_display.replace("\\_", "_"),
                f"{r.file_size_mib:.1f}",
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


def write_csv_out(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "size_gib",
                "sparsity_pct",
                "best_ratio",
                "implementation",
                "file_size_mib",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    f"{r.size_gib:.6f}",
                    f"{r.sparsity_pct:.3f}",
                    f"{r.best_ratio:.6f}",
                    r.implementation_display.replace("\\_", "_"),
                    f"{r.file_size_mib:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset summary table")
    parser.add_argument(
        "--dataset-inventory",
        type=Path,
        default=None,
        help="Path to dataset_inventory.csv (default: results/dataset_inventory.csv)",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
        help="Path to aggregated statistics.csv (default: results/aggregated/statistics.csv)",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_script(Path(__file__))
    inv_path = args.dataset_inventory or (
        repo_root / "results" / "dataset_inventory.csv"
    )
    stats_path = args.statistics or (
        repo_root / "results" / "aggregated" / "statistics.csv"
    )

    inv = pd.read_csv(inv_path)
    stats = pd.read_csv(stats_path)
    rows = build_rows(inv, stats)

    out_tex = (
        repo_root / "paper" / "generated" / "tables" / "Table1_dataset_summary.tex"
    )
    out_txt = (
        repo_root
        / "paper"
        / "generated"
        / "tables_ascii"
        / "Table1_dataset_summary.txt"
    )
    out_csv = (
        repo_root / "paper" / "generated" / "tables_csv" / "Table1_dataset_summary.csv"
    )

    write_latex(rows, out_tex)
    write_ascii(rows, out_txt)
    write_csv_out(rows, out_csv)

    print("✓ Generated public dataset summary table outputs:")
    print(f"  LaTeX: {out_tex}")
    print(f"  ASCII: {out_txt}")
    print(f"  CSV:   {out_csv}")


if __name__ == "__main__":
    main()
