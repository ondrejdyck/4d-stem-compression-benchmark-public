#!/usr/bin/env python3
"""Generate the Methods dataset table in LaTeX/ASCII/CSV.

Outputs (synchronized):
- LaTeX:  paper/generated/tables/table_datasets.tex
- ASCII:  paper/generated/tables_ascii/table_datasets.txt
- CSV:    paper/generated/tables_csv/table_datasets.csv

Input (source of truth):
- results/dataset_inventory.csv

Notes
-----
- Dataset sizes are reported in binary units (GiB/MiB): 1 GiB = 1024^3 bytes,
  1 MiB = 1024^2 bytes.
- Sparsity is the fraction of values exactly equal to zero.
- Descriptions are intentionally hard-coded (they are narrative, not computed).
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
    shape: str
    size_display: str
    sparsity_pct: float
    description: str


def _repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _size_display(uncompressed_bytes: int) -> str:
    gib = uncompressed_bytes / (1024**3)
    if gib < 0.1:
        mib = uncompressed_bytes / (1024**2)
        return f"{mib:.1f} MiB"
    return f"{gib:.1f} GiB"


def build_rows(inv: pd.DataFrame) -> list[Row]:
    inv = inv.set_index("dataset_id")

    order = [
        ("4D_EELS", "4D_EELS"),
        ("4D_Diff", "4D_Diff"),
        ("4D_Diff-2x2-binning", "4D_Diff-2x2"),
        ("4D_Diff-4x4-binning", "4D_Diff-4x4"),
        ("3D_EELS", "3D_EELS"),
    ]

    descriptions = {
        "4D_EELS": "Full 4D EELS spectrum imaging",
        "4D_Diff": "4D STEM diffraction (unbinned)",
        "4D_Diff-2x2": "4D STEM diffraction (2$\\times$2 binned)",
        "4D_Diff-4x4": "4D STEM diffraction (4$\\times$4 binned)",
        "3D_EELS": "Y-summed EELS spectrum image",
    }

    rows: list[Row] = []
    for dataset_id, display_name in order:
        if dataset_id not in inv.index:
            raise ValueError(f"Dataset not found in inventory: {dataset_id}")

        uncompressed_bytes = int(inv.loc[dataset_id, "uncompressed_bytes"])
        sparsity = float(inv.loc[dataset_id, "sparsity_fraction"]) * 100
        shape = str(inv.loc[dataset_id, "shape"]).strip('"')

        rows.append(
            Row(
                dataset=display_name,
                shape=shape,
                size_display=_size_display(uncompressed_bytes),
                sparsity_pct=sparsity,
                description=descriptions[display_name],
            )
        )

    return rows


def write_csv_out(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "shape", "size", "sparsity_pct", "description"])
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    r.shape,
                    r.size_display,
                    f"{r.sparsity_pct:.3f}",
                    r.description,
                ]
            )


def write_ascii(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Dataset", "Shape", "Size", "Sparsity", "Description"]
    table_rows: list[list[str]] = []
    for r in rows:
        table_rows.append(
            [
                r.dataset,
                r.shape,
                r.size_display,
                f"{r.sparsity_pct:.1f}%",
                r.description,
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
    lines.append("\\caption{Datasets used for compression benchmarking}")
    lines.append("\\label{tab:datasets}")
    lines.append("\\begin{tabular}{lllrl}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Dataset} & \\textbf{Shape} & \\textbf{Size} & \\textbf{Sparsity} & \\textbf{Description} \\\\"
    )
    lines.append("\\hline")

    for r in rows:
        ds = _latex_escape(r.dataset)
        sparsity = f"{r.sparsity_pct:.1f}\\%"
        lines.append(
            f"{ds} & {r.shape} & {r.size_display} & {sparsity} & {r.description} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Methods dataset table")
    parser.add_argument(
        "--dataset-inventory",
        type=Path,
        default=None,
        help="Path to dataset_inventory.csv (default: results/dataset_inventory.csv)",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_script(Path(__file__))
    inv_path = args.dataset_inventory or (
        repo_root / "results" / "dataset_inventory.csv"
    )

    inv = pd.read_csv(inv_path)
    rows = build_rows(inv)

    out_tex = repo_root / "paper" / "generated" / "tables" / "table_datasets.tex"
    out_txt = repo_root / "paper" / "generated" / "tables_ascii" / "table_datasets.txt"
    out_csv = repo_root / "paper" / "generated" / "tables_csv" / "table_datasets.csv"

    write_latex(rows, out_tex)
    write_ascii(rows, out_txt)
    write_csv_out(rows, out_csv)

    print("✓ Generated Methods dataset table outputs:")
    print(f"  LaTeX: {out_tex}")
    print(f"  ASCII: {out_txt}")
    print(f"  CSV:   {out_csv}")


if __name__ == "__main__":
    main()
