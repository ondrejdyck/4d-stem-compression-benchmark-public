#!/usr/bin/env python3
"""Build dataset inventory for paper tables.

This script computes dataset metadata directly from the local EMD (.emd) files
used in the benchmark, including:

- shape and dtype of the primary datacube
- uncompressed byte size (based on shape * dtype.itemsize)
- exact sparsity (fraction of values exactly equal to zero)
- global maximum value (useful for diagnosing effective bit depth)

The resulting CSV is intended to be committed alongside aggregated benchmark
statistics so that paper tables can be generated from code rather than
hand-copied values.

Notes
-----
- The raw datasets themselves are not suitable for public release in git due to
  size. This script serves as an audit tool to regenerate the metadata when the
  datasets are available locally.
- Sparsity is defined as the fraction of values exactly equal to zero. This is
  appropriate for lossless compression comparisons.

Default input dataset path inside EMD files:
  version_1/data/datacubes/datacube_000/data

Default output:
  results/dataset_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EMD_DATASET_PATH_DEFAULT = "version_1/data/datacubes/datacube_000/data"


@dataclass(frozen=True)
class DatasetInventoryRow:
    dataset_id: str
    filename: str
    hdf5_path: str
    dtype: str
    itemsize_bytes: int
    shape: tuple[int, int, int, int]
    n_elements: int
    file_size_bytes: int
    uncompressed_bytes: int
    uncompressed_gb: float
    uncompressed_mb: float
    zero_count: int
    sparsity_fraction: float
    max_value: float
    max_location: tuple[int, int, int, int]
    max_bits_int: int | None
    count_gt_4095: int
    count_gt_16383: int


def _repo_root_from_script(script_path: Path) -> Path:
    # .../implementation/src/paper_artifacts/datasets/build_dataset_inventory.py
    # parents: datasets -> paper_artifacts -> src -> implementation -> repo_root
    return script_path.resolve().parents[4]


def _canonical_dataset_id_from_stem(stem: str) -> str:
    """Map local filenames to canonical dataset IDs used in the manuscript."""
    mapping = {
        "3D_EELS": "3D_EELS",
        "4D_EELS": "4D_EELS",
        "4D_Diff": "4D_Diff",
        "4D_Diff-2x2-binning": "4D_Diff-2x2-binning",
        "4D_Diff-4x4-binning": "4D_Diff-4x4-binning",
    }
    return mapping.get(stem, stem)


def _int_bits_needed(max_int: int) -> int:
    if max_int < 0:
        raise ValueError("Expected non-negative integer for bit calculation")
    if max_int == 0:
        return 1
    return int(math.ceil(math.log2(max_int + 1)))


def compute_inventory_for_file(file_path: Path, hdf5_path: str) -> DatasetInventoryRow:
    with h5py.File(file_path, "r") as h:
        dset = h[hdf5_path]

        if len(dset.shape) != 4:
            raise ValueError(
                f"Expected 4D datacube at {hdf5_path}, got shape={dset.shape}"
            )

        shape = tuple(int(x) for x in dset.shape)  # type: ignore[assignment]
        dtype = dset.dtype
        itemsize = int(dtype.itemsize)
        n_elements = int(np.prod(shape))

        file_size_bytes = int(file_path.stat().st_size)
        uncompressed_bytes = n_elements * itemsize
        uncompressed_gb = uncompressed_bytes / 1e9
        uncompressed_mb = uncompressed_bytes / 1e6

        # Stream through the first scan dimension to avoid loading the full datacube.
        n0, n1, ny, nx = shape
        zero_count = 0
        count_gt_4095 = 0
        count_gt_16383 = 0

        max_value = float("-inf")
        max_location: tuple[int, int, int, int] = (0, 0, 0, 0)

        for i in range(n0):
            # Read a block for fixed scan index i: shape (n1, ny, nx)
            block = dset[i, :, :, :]

            # Exact zeros (works for int and float)
            zero_count += int(np.sum(block == 0))

            # Threshold counts (useful for diagnosing effective dynamic range)
            count_gt_4095 += int(np.sum(block > 4095))
            count_gt_16383 += int(np.sum(block > 16383))

            bmax = float(np.max(block))
            if bmax > max_value:
                # Find the location of the maximum within this block
                idx = np.unravel_index(int(np.argmax(block)), block.shape)
                # idx = (j, y, x)
                max_value = bmax
                max_location = (i, int(idx[0]), int(idx[1]), int(idx[2]))

        sparsity_fraction = zero_count / n_elements

        max_bits_int: int | None = None
        if np.issubdtype(dtype, np.integer):
            max_bits_int = _int_bits_needed(int(max_value))

        return DatasetInventoryRow(
            dataset_id=_canonical_dataset_id_from_stem(file_path.stem),
            filename=file_path.name,
            hdf5_path=hdf5_path,
            dtype=str(dtype),
            itemsize_bytes=itemsize,
            shape=shape,
            n_elements=n_elements,
            file_size_bytes=file_size_bytes,
            uncompressed_bytes=uncompressed_bytes,
            uncompressed_gb=uncompressed_gb,
            uncompressed_mb=uncompressed_mb,
            zero_count=zero_count,
            sparsity_fraction=sparsity_fraction,
            max_value=max_value,
            max_location=max_location,
            max_bits_int=max_bits_int,
            count_gt_4095=count_gt_4095,
            count_gt_16383=count_gt_16383,
        )


def write_csv(rows: list[DatasetInventoryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset_id",
        "filename",
        "hdf5_path",
        "dtype",
        "itemsize_bytes",
        "shape",
        "n_elements",
        "file_size_bytes",
        "uncompressed_bytes",
        "uncompressed_gb",
        "uncompressed_mb",
        "zero_count",
        "sparsity_fraction",
        "max_value",
        "max_location",
        "max_bits_int",
        "count_gt_4095",
        "count_gt_16383",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset_id": r.dataset_id,
                    "filename": r.filename,
                    "hdf5_path": r.hdf5_path,
                    "dtype": r.dtype,
                    "itemsize_bytes": r.itemsize_bytes,
                    "shape": str(r.shape),
                    "n_elements": r.n_elements,
                    "file_size_bytes": r.file_size_bytes,
                    "uncompressed_bytes": r.uncompressed_bytes,
                    "uncompressed_gb": f"{r.uncompressed_gb:.6f}",
                    "uncompressed_mb": f"{r.uncompressed_mb:.3f}",
                    "zero_count": r.zero_count,
                    "sparsity_fraction": f"{r.sparsity_fraction:.6f}",
                    "max_value": f"{r.max_value:.6f}",
                    "max_location": str(r.max_location),
                    "max_bits_int": "" if r.max_bits_int is None else r.max_bits_int,
                    "count_gt_4095": r.count_gt_4095,
                    "count_gt_16383": r.count_gt_16383,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dataset inventory metadata from local EMD files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .emd files (default: implementation/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/dataset_inventory.csv)",
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default=EMD_DATASET_PATH_DEFAULT,
        help=f"HDF5 dataset path inside EMD files (default: {EMD_DATASET_PATH_DEFAULT})",
    )
    args = parser.parse_args()

    script_path = Path(__file__)
    repo_root = _repo_root_from_script(script_path)

    data_dir = args.data_dir or (repo_root / "implementation" / "data")
    output_path = args.output or (repo_root / "results" / "dataset_inventory.csv")

    emd_files = sorted(data_dir.glob("*.emd"))
    if not emd_files:
        raise FileNotFoundError(f"No .emd files found in: {data_dir}")

    rows: list[DatasetInventoryRow] = []
    print(f"Reading {len(emd_files)} EMD files from: {data_dir}")
    for fp in emd_files:
        print(f"  - {fp.name}")
        rows.append(compute_inventory_for_file(fp, args.hdf5_path))

    # Sort in a stable, human-friendly order
    rows.sort(key=lambda r: r.dataset_id)

    write_csv(rows, output_path)
    print(f"\n✓ Wrote dataset inventory: {output_path}")


if __name__ == "__main__":
    main()
