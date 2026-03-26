#!/usr/bin/env python3
"""
Compression Benchmark for 4D STEM Data

Tests various compression strategies on MIB files to determine optimal
storage and access patterns for large 4D datasets.
"""

import numpy as np
import h5py
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import sparse
import gzip
import pickle
import pandas as pd
import json
import psutil

# Advanced compression libraries
try:
    import hdf5plugin

    HAS_HDF5PLUGIN = True
except ImportError:
    HAS_HDF5PLUGIN = False
    print(
        "Warning: hdf5plugin not available. Advanced compression methods will be skipped."
    )


# EMD/HDF5 Loading Functions
def load_emd(filepath):
    """Load EMD 1.0 format 4D STEM data.

    Parameters
    ----------
    filepath : str or Path
        Path to .emd file

    Returns
    -------
    np.ndarray
        4D array with shape (scan_y, scan_x, det_y, det_x)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        # EMD 1.0 standard structure
        try:
            data = f["version_1/data/datacubes/datacube_000/data"][:]
            return data
        except KeyError:
            raise ValueError(
                f"Not a valid EMD 1.0 file. Expected path: version_1/data/datacubes/datacube_000/data"
            )


def load_h5_generic(filepath, dataset_path=None):
    """Load generic HDF5 4D data.

    Parameters
    ----------
    filepath : str or Path
        Path to .h5/.hdf5 file
    dataset_path : str, optional
        Path to dataset within HDF5. If None, auto-detect.

    Returns
    -------
    np.ndarray
        4D array
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        if dataset_path:
            return f[dataset_path][:]

        # Try common paths
        common_paths = [
            "version_1/data/datacubes/datacube_000/data",  # EMD 1.0
            "data",
            "dataset",
            "4DSTEM_experiment/data/datacubes/datacube_0/data",  # py4DSTEM
        ]

        for path in common_paths:
            try:
                data = f[path][:]
                if len(data.shape) == 4:
                    print(f"Found 4D dataset at: {path}")
                    return data
            except KeyError:
                continue

        # List available datasets if nothing found
        print("Could not auto-detect dataset. Available datasets:")

        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(print_datasets)
        raise ValueError("No 4D dataset found in file")


def analyze_data_sparsity(data_4d):
    """Analyze sparsity and distribution of 4D data"""
    print("=== Data Sparsity Analysis ===")

    # Flatten for analysis
    flat_data = data_4d.ravel()

    # Basic statistics
    print(f"Shape: {data_4d.shape}")
    print(f"Total elements: {flat_data.size:,}")
    print(f"Data type: {data_4d.dtype}")
    print(f"Memory size: {data_4d.nbytes / (1024**3):.3f} GiB")

    # Value distribution
    unique_vals, counts = np.unique(flat_data, return_counts=True)
    zero_count = counts[0] if unique_vals[0] == 0 else 0

    print(f"\nValue Distribution:")
    print(f"  Min: {flat_data.min()}")
    print(f"  Max: {flat_data.max()}")
    print(f"  Mean: {flat_data.mean():.2f}")
    print(f"  Std: {flat_data.std():.2f}")
    print(f"  Zeros: {zero_count:,} ({100 * zero_count / flat_data.size:.1f}%)")
    print(f"  Non-zeros: {flat_data.size - zero_count:,}")
    print(f"  Unique values: {len(unique_vals):,}")

    # Show value histogram for small values
    small_vals = flat_data[flat_data <= 50]
    if len(small_vals) > 0:
        print(
            f"  Values ≤ 50: {len(small_vals):,} ({100 * len(small_vals) / flat_data.size:.1f}%)"
        )

    return {
        "sparsity": zero_count / flat_data.size,
        "unique_values": len(unique_vals),
        "max_value": flat_data.max(),
        "mean_value": flat_data.mean(),
    }


def benchmark_hdf5_compression(data_4d, output_dir):
    """Test HDF5 compression algorithms"""
    print("\n=== HDF5 Compression Benchmark ===")

    results = {}
    original_size = data_4d.nbytes

    # Test different compression algorithms
    compression_methods = {
        "none": None,
        "gzip_1": ("gzip", 1),
        "gzip_6": ("gzip", 6),
        "gzip_9": ("gzip", 9),
        "lzf": ("lzf", None),
        "szip": ("szip", None),
    }

    # Add advanced methods if hdf5plugin is available
    if HAS_HDF5PLUGIN:
        print("Adding advanced compression methods (Blosc, LZ4, Bitshuffle)...")
        compression_methods.update(
            {
                "blosc_blosclz": (
                    hdf5plugin.Blosc(
                        cname="blosclz", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                    None,
                ),
                "blosc_lz4": (
                    hdf5plugin.Blosc(
                        cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                    None,
                ),
                "blosc_lz4hc": (
                    hdf5plugin.Blosc(
                        cname="lz4hc", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                    None,
                ),
                "blosc_zlib": (
                    hdf5plugin.Blosc(
                        cname="zlib", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                    None,
                ),
                "blosc_zstd": (
                    hdf5plugin.Blosc(
                        cname="zstd", clevel=3, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                    None,
                ),
                "lz4_hdf5": (hdf5plugin.LZ4(nbytes=0), None),
                "bitshuffle_lz4": (hdf5plugin.Bitshuffle(nelems=0, lz4=True), None),
            }
        )
    else:
        print("Skipping advanced compression methods (hdf5plugin not installed)")

    # Generate chunk sizes based on actual data shape
    sy, sx, qy, qx = data_4d.shape
    chunk_sizes = [
        (min(32, sy), min(32, sx), min(256, qy), min(256, qx)),  # Real-space optimized
        (min(16, sy), min(16, sx), min(128, qy), min(128, qx)),  # Balanced
        (1, 1, qy, qx),  # Single frame chunks
    ]

    for chunk_name, chunk_size in zip(
        ["real_space", "balanced", "single_frame"], chunk_sizes
    ):
        print(f"\nChunk size: {chunk_size}")

        for method_name, method_config in compression_methods.items():
            try:
                filename = output_dir / f"test_{chunk_name}_{method_name}.h5"

                start_time = time.time()

                with h5py.File(filename, "w") as f:
                    if method_config is None:
                        # No compression
                        dataset = f.create_dataset(
                            "data", data=data_4d, chunks=chunk_size
                        )
                    else:
                        compression, opts = method_config

                        # Check if it's an hdf5plugin filter object
                        if hasattr(compression, "__class__") and hasattr(
                            compression, "filter_id"
                        ):
                            # hdf5plugin filter - use as keyword arguments
                            dataset = f.create_dataset(
                                "data", data=data_4d, chunks=chunk_size, **compression
                            )
                        elif opts is not None:
                            # Standard HDF5 with options
                            dataset = f.create_dataset(
                                "data",
                                data=data_4d,
                                compression=compression,
                                compression_opts=opts,
                                chunks=chunk_size,
                            )
                        else:
                            # Standard HDF5 without options
                            dataset = f.create_dataset(
                                "data",
                                data=data_4d,
                                compression=compression,
                                chunks=chunk_size,
                            )

                write_time = time.time() - start_time
                file_size = filename.stat().st_size
                compression_ratio = original_size / file_size

                # Test read speed - FULL DATASET READ with proper measurement
                # Single read operation - variability will be measured across multiple benchmark runs
                #
                # NOTE: actual_io_bytes tracks real disk I/O (from /proc/self/io on Linux).
                # This is typically 0 because the OS caches file data in RAM after first access.
                # Our benchmarks measure cache-to-RAM performance (realistic for iterative analysis),
                # not cold disk-to-RAM performance. The actual_io_bytes=0 confirms all methods
                # are compared fairly under the same caching conditions.

                # Get I/O stats before read
                process = psutil.Process()
                io_before = process.io_counters()

                # Open file and read full dataset
                start_time = time.time()
                with h5py.File(filename, "r") as f:
                    # Separate file open from data read
                    dset = f["data"]

                    # Pure data read timing
                    read_start = time.time()
                    data_read = dset[:]
                    read_time = time.time() - read_start

                total_elapsed = time.time() - start_time

                # Get I/O stats after read
                io_after = process.io_counters()

                # Store measurements
                bytes_read = data_read.nbytes
                actual_io_bytes = io_after.read_bytes - io_before.read_bytes

                # Calculate throughput
                write_throughput_gbs = (original_size / (1024**3)) / write_time
                read_throughput_gbs = (bytes_read / (1024**3)) / read_time

                results[f"{chunk_name}_{method_name}"] = {
                    "file_size_mb": file_size / (1024**2),
                    "compression_ratio": compression_ratio,
                    "write_time": write_time,
                    "read_time": read_time,
                    "chunk_size": chunk_size,
                    "bytes_read": bytes_read,
                    "actual_io_bytes": actual_io_bytes,
                    "write_throughput_gbs": write_throughput_gbs,
                    "read_throughput_gbs": read_throughput_gbs,
                }

                print(
                    f"  {method_name:8s}: {file_size / (1024**2):6.1f} MiB "
                    f"({compression_ratio:4.1f}x) "
                    f"W:{write_time:5.1f}s ({write_throughput_gbs:.2f} GiB/s) "
                    f"R:{read_time:5.2f}s ({read_throughput_gbs:.2f} GiB/s)"
                )

                # Clean up
                filename.unlink()

            except Exception as e:
                print(f"  {method_name:8s}: FAILED ({e})")

    return results


def benchmark_sparse_storage(data_4d, output_dir):
    """Test sparse matrix storage"""
    print("\n=== Sparse Storage Benchmark ===")

    original_size = data_4d.nbytes
    results = {}

    # Test storing as 2D sparse matrices (frame by frame)
    sparse_frames = []
    start_time = time.time()

    for sy in range(data_4d.shape[0]):
        for sx in range(data_4d.shape[1]):
            frame = data_4d[sy, sx, :, :]
            # Convert to little-endian uint16 for sparse matrix compatibility
            frame_le = frame.astype(np.uint16)
            sparse_frame = sparse.csr_matrix(frame_le)
            sparse_frames.append(sparse_frame)

    creation_time = time.time() - start_time

    # Save to file
    filename = output_dir / "sparse_frames.pkl"
    start_time = time.time()
    with open(filename, "wb") as f:
        pickle.dump(sparse_frames, f)
    save_time = time.time() - start_time

    file_size = filename.stat().st_size
    compression_ratio = original_size / file_size

    # Test loading speed
    start_time = time.time()
    with open(filename, "rb") as f:
        loaded_frames = pickle.load(f)
    # Convert a few frames back to dense
    _ = loaded_frames[0].toarray()
    if len(loaded_frames) > 100:
        _ = loaded_frames[100].toarray()
    load_time = time.time() - start_time

    results["sparse_csr"] = {
        "file_size_mb": file_size / (1024**2),
        "compression_ratio": compression_ratio,
        "creation_time": creation_time,
        "save_time": save_time,
        "load_time": load_time,
    }

    print(f"Sparse CSR: {file_size / (1024**2):6.1f} MiB ({compression_ratio:4.1f}x)")
    print(
        f"  Create: {creation_time:.1f}s, Save: {save_time:.1f}s, Load: {load_time:.1f}s"
    )

    # Clean up
    filename.unlink()

    return results


def benchmark_custom_compression(data_4d, output_dir):
    """Test custom compression strategies"""
    print("\n=== Custom Compression Benchmark ===")

    original_size = data_4d.nbytes
    results = {}

    # Strategy 1: uint8 + overflow map (for values mostly < 255)
    start_time = time.time()

    # Convert to uint8, track overflow
    data_uint8 = np.clip(data_4d, 0, 254).astype(np.uint8)
    overflow_mask = data_4d >= 255
    overflow_values = data_4d[overflow_mask]
    overflow_coords = np.where(overflow_mask)

    # Mark overflow pixels as 255 in uint8 array
    data_uint8[overflow_mask] = 255

    conversion_time = time.time() - start_time

    # Save compressed version
    filename_main = output_dir / "custom_uint8.npy"
    filename_overflow = output_dir / "custom_overflow.npz"

    start_time = time.time()
    np.save(filename_main, data_uint8)
    np.savez_compressed(
        filename_overflow,
        coords=np.column_stack(overflow_coords),
        values=overflow_values,
    )
    save_time = time.time() - start_time

    file_size = filename_main.stat().st_size + filename_overflow.stat().st_size
    compression_ratio = original_size / file_size

    print(
        f"uint8 + overflow: {file_size / (1024**2):6.1f} MiB ({compression_ratio:4.1f}x)"
    )
    print(
        f"  Overflow pixels: {len(overflow_values):,} ({100 * len(overflow_values) / data_4d.size:.2f}%)"
    )
    print(f"  Convert: {conversion_time:.1f}s, Save: {save_time:.1f}s")

    results["uint8_overflow"] = {
        "file_size_mb": file_size / (1024**2),
        "compression_ratio": compression_ratio,
        "conversion_time": conversion_time,
        "save_time": save_time,
        "overflow_fraction": len(overflow_values) / data_4d.size,
    }

    # Clean up
    filename_main.unlink()
    filename_overflow.unlink()

    # Strategy 2: Simple gzip compression
    start_time = time.time()
    filename_gz = output_dir / "simple_gzip.npy.gz"
    with gzip.open(filename_gz, "wb") as f:
        np.save(f, data_4d)
    gzip_time = time.time() - start_time

    gzip_size = filename_gz.stat().st_size
    gzip_ratio = original_size / gzip_size

    print(f"Simple gzip: {gzip_size / (1024**2):6.1f} MiB ({gzip_ratio:4.1f}x)")
    print(f"  Time: {gzip_time:.1f}s")

    results["simple_gzip"] = {
        "file_size_mb": gzip_size / (1024**2),
        "compression_ratio": gzip_ratio,
        "compression_time": gzip_time,
    }

    # Clean up
    filename_gz.unlink()

    return results


def create_summary_plot(all_results, output_dir):
    """Create summary plots of compression results"""
    print("\n=== Creating Summary Plots ===")

    # Extract data for plotting
    methods = []
    ratios = []
    sizes_mb = []

    for method, data in all_results.items():
        methods.append(method)
        ratios.append(data["compression_ratio"])
        sizes_mb.append(data["file_size_mb"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Compression ratio plot
    bars1 = ax1.bar(range(len(methods)), ratios)
    ax1.set_xlabel("Compression Method")
    ax1.set_ylabel("Compression Ratio")
    ax1.set_title("Compression Ratios (Higher = Better)")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, ratio in zip(bars1, ratios):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
        )

    # File size plot
    bars2 = ax2.bar(range(len(methods)), sizes_mb)
    ax2.set_xlabel("Compression Method")
    ax2.set_ylabel("File Size (MiB)")
    ax2.set_title("Compressed File Sizes (Lower = Better)")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, size in zip(bars2, sizes_mb):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{size:.0f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "compression_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


def save_results_to_csv(results, output_dir, dataset_name):
    """Save benchmark results to CSV for later analysis

    Parameters
    ----------
    results : dict
        Results dictionary from run_benchmark()
    output_dir : Path
        Directory to save CSV and metadata
    dataset_name : str
        Name of the dataset

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results into rows
    rows = []
    for method_name, method_data in results["compression_results"].items():
        row = {
            "dataset": dataset_name,
            "method": method_name,
            "compression_ratio": method_data["compression_ratio"],
            "file_size_mb": method_data["file_size_mb"],
        }

        # Add timing data (handle different key names)
        write_time = method_data.get(
            "write_time",
            method_data.get("save_time", method_data.get("compression_time", None)),
        )
        read_time = method_data.get("read_time", method_data.get("load_time", None))

        row["write_time"] = write_time
        row["read_time"] = read_time

        # Add throughput data if available
        if "write_throughput_gbs" in method_data:
            row["write_throughput_gbs"] = method_data["write_throughput_gbs"]
        if "read_throughput_gbs" in method_data:
            row["read_throughput_gbs"] = method_data["read_throughput_gbs"]
        if "bytes_read" in method_data:
            row["bytes_read"] = method_data["bytes_read"]
        if "actual_io_bytes" in method_data:
            row["actual_io_bytes"] = method_data["actual_io_bytes"]

        # Add chunk size if available
        if "chunk_size" in method_data:
            row["chunk_size"] = str(method_data["chunk_size"])
        else:
            row["chunk_size"] = None

        # Add other metadata
        if "creation_time" in method_data:
            row["creation_time"] = method_data["creation_time"]
        if "conversion_time" in method_data:
            row["conversion_time"] = method_data["conversion_time"]

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    csv_file = output_dir / "benchmark_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Results saved to: {csv_file}")

    # Save metadata separately
    metadata = {
        "dataset_name": results["dataset_name"],
        "data_shape": list(results["data_shape"]),  # Convert tuple to list for JSON
        "data_dtype": results["data_dtype"],
        "original_size_mb": results["original_size_mb"],
        "sparsity": results["sparsity_info"]["sparsity"],
        "unique_values": results["sparsity_info"]["unique_values"],
        "max_value": int(results["sparsity_info"]["max_value"]),
        "mean_value": float(results["sparsity_info"]["mean_value"]),
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")

    return df


def run_benchmark(
    data_4d, output_dir, dataset_name="dataset", save_csv=True, create_plots=False
):
    """
    Run compression benchmark on 4D STEM data.

    Parameters
    ----------
    data_4d : np.ndarray
        4D STEM dataset with shape (scan_y, scan_x, det_y, det_x)
    output_dir : Path or str
        Directory to save results
    dataset_name : str, optional
        Name of the dataset for labeling results
    save_csv : bool, optional
        Save results to CSV file (default: True)
    create_plots : bool, optional
        Create summary plots (default: False, do in notebook instead)

    Returns
    -------
    dict
        Dictionary containing all benchmark results and metadata
    """
    print(f"4D STEM Compression Benchmark: {dataset_name}")
    print("=" * 50)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze sparsity
    sparsity_info = analyze_data_sparsity(data_4d)

    # Run benchmarks
    all_results = {}

    # HDF5 compression
    hdf5_results = benchmark_hdf5_compression(data_4d, output_dir)
    all_results.update(hdf5_results)

    # Sparse storage
    sparse_results = benchmark_sparse_storage(data_4d, output_dir)
    all_results.update(sparse_results)

    # Custom compression
    custom_results = benchmark_custom_compression(data_4d, output_dir)
    all_results.update(custom_results)

    # Create summary
    print("\n" + "=" * 70)
    print("COMPRESSION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Size (MiB)':<12} {'Ratio':<8} {'Notes'}")
    print("-" * 70)

    original_size_mb = data_4d.nbytes / (1024**2)
    print(f"{'Original':<25} {original_size_mb:<12.1f} {'1.0x':<8} {'Uncompressed'}")

    # Sort by compression ratio
    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["compression_ratio"], reverse=True
    )

    for method, data in sorted_results:
        notes = ""
        if "gzip" in method:
            notes = "Standard compression"
        elif "sparse" in method:
            notes = f"Sparse matrix"
        elif "uint8" in method:
            notes = f"8-bit + {data.get('overflow_fraction', 0) * 100:.1f}% overflow"

        print(
            f"{method:<25} {data['file_size_mb']:<12.1f} {data['compression_ratio']:<8.1f} {notes}"
        )

    # Create plots (optional, usually done in notebook)
    if create_plots:
        create_summary_plot(all_results, output_dir)

    # Save detailed results (text file for reference)
    results_file = output_dir / f"{dataset_name}_detailed_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Detailed Compression Benchmark Results: {dataset_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Original data: {data_4d.shape}, {data_4d.dtype}\n")
        f.write(f"Original size: {original_size_mb:.1f} MiB\n")
        f.write(f"Sparsity: {sparsity_info['sparsity'] * 100:.1f}% zeros\n\n")

        for method, data in sorted_results:
            f.write(f"{method}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"\n✓ Detailed results saved to: {results_file}")
    if create_plots:
        print(f"✓ Plots saved to: {output_dir / 'compression_benchmark.png'}")

    # Return comprehensive results
    results = {
        "dataset_name": dataset_name,
        "data_shape": data_4d.shape,
        "data_dtype": str(data_4d.dtype),
        "original_size_mb": original_size_mb,
        "sparsity_info": sparsity_info,
        "compression_results": all_results,
        "sorted_results": sorted_results,
    }

    # Save to CSV if requested
    if save_csv:
        save_results_to_csv(results, output_dir, dataset_name)

    return results


def main():
    """Main benchmarking function - loads the default local dataset or smoke fixture."""
    print("4D STEM Data Compression Benchmark")
    print("=" * 50)

    # Setup - use relative paths from implementation directory
    data_dir = Path(__file__).parent.parent / "data"
    fixture_dir = Path(__file__).parent.parent / "fixtures"
    output_dir = Path(__file__).parent.parent / "results"

    # Load the default EMD dataset if available; otherwise fall back to the
    # bundled synthetic smoke fixture so the public repo still runs end-to-end.
    emd_file = data_dir / "4D_EELS.emd"
    if not emd_file.exists():
        emd_file = fixture_dir / "smoke_test.emd"

    if not emd_file.exists():
        print(f"ERROR: Could not find {emd_file}")
        print(f"Looking in: {data_dir.absolute()}")
        print(f"Also checked: {fixture_dir.absolute()}")
        print("\nAvailable files:")
        if data_dir.exists():
            for f in data_dir.glob("*.emd"):
                print(f"  {f.name}")
        if fixture_dir.exists():
            for f in fixture_dir.glob("*.emd"):
                print(f"  {f.name}")
        return

    print(f"Loading: {emd_file.name}")
    dataset_name = "4D_EELS" if emd_file.parent == data_dir else "smoke_test"
    if emd_file.parent == fixture_dir:
        print("Using bundled smoke fixture.")
    print("This may take a moment...")

    # Load data
    start_time = time.time()
    data_4d = load_emd(emd_file)
    load_time = time.time() - start_time

    print(f"Loaded in {load_time:.1f}s")

    # Run benchmark using the new function
    results = run_benchmark(data_4d, output_dir, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
