#!/usr/bin/env python3
"""
Read Speed Diagnostic Tool

Systematically tests read performance to identify the source of the 13× speed difference
between EELS and diffraction datasets.

Tests performed:
1. Storage device check (same disk?)
2. File size and location
3. Cold cache vs warm cache reads
4. Sequential read patterns
5. HDF5 chunk cache effects

Usage:
    python diagnose_read_speed.py
"""

import h5py
import numpy as np
import time
import os
import subprocess
from pathlib import Path
import json


def get_file_device(filepath):
    """Get the device where a file is stored."""
    try:
        result = subprocess.run(
            ["df", str(filepath)], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            device_info = lines[1].split()
            return {
                "device": device_info[0],
                "size": device_info[1],
                "used": device_info[2],
                "available": device_info[3],
                "use_percent": device_info[4],
                "mountpoint": device_info[5],
            }
    except Exception as e:
        return {"error": str(e)}
    return None


def get_file_info(filepath):
    """Get detailed file information."""
    stat = os.stat(filepath)
    return {
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024**2),
        "size_gb": stat.st_size / (1024**3),
        "blocks": stat.st_blocks,
        "block_size": stat.st_blksize,
    }


def clear_cache():
    """Attempt to clear file system cache (requires sudo)."""
    try:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            timeout=5,
        )
        return True
    except Exception as e:
        print(f"  ⚠ Could not clear cache (requires sudo): {e}")
        return False


def test_read_speed(filepath, dataset_path, test_name="Test", clear_cache_first=False):
    """
    Test read speed for an HDF5 dataset.

    Returns dict with timing information.
    """
    if clear_cache_first:
        print(f"  Clearing cache...")
        clear_cache()
        time.sleep(1)  # Let system settle

    results = {}

    # Test 1: Open file
    t0 = time.time()
    f = h5py.File(filepath, "r")
    t1 = time.time()
    results["open_time"] = t1 - t0

    # Test 2: Access dataset metadata
    t0 = time.time()
    dset = f[dataset_path]
    shape = dset.shape
    dtype = dset.dtype
    chunks = dset.chunks
    t1 = time.time()
    results["metadata_time"] = t1 - t0
    results["shape"] = shape
    results["dtype"] = str(dtype)
    results["chunks"] = chunks

    # Test 3: Read entire dataset
    t0 = time.time()
    data = dset[:]
    t1 = time.time()
    results["read_time"] = t1 - t0
    results["data_size_gb"] = data.nbytes / (1024**3)
    results["throughput_gbs"] = results["data_size_gb"] / results["read_time"]

    # Test 4: Read again (should be cached)
    t0 = time.time()
    data2 = dset[:]
    t1 = time.time()
    results["read_time_cached"] = t1 - t0
    results["throughput_gbs_cached"] = (
        results["data_size_gb"] / results["read_time_cached"]
    )

    f.close()

    return results


def test_partial_reads(filepath, dataset_path, n_samples=10):
    """Test reading random subsets of data."""
    results = {}

    with h5py.File(filepath, "r") as f:
        dset = f[dataset_path]
        shape = dset.shape

        # Test reading single frames
        frame_times = []
        for i in range(min(n_samples, shape[0])):
            t0 = time.time()
            frame = dset[i, 0, :, :]
            t1 = time.time()
            frame_times.append(t1 - t0)

        results["single_frame_mean_time"] = np.mean(frame_times)
        results["single_frame_std_time"] = np.std(frame_times)

        # Test reading slices
        slice_times = []
        for i in range(min(n_samples, shape[0] // 10)):
            idx = i * (shape[0] // n_samples)
            t0 = time.time()
            slice_data = dset[idx : idx + 10, :, :, :]
            t1 = time.time()
            slice_times.append(t1 - t0)

        results["slice_mean_time"] = np.mean(slice_times)
        results["slice_std_time"] = np.std(slice_times)

    return results


def main():
    print("=" * 100)
    print("READ SPEED DIAGNOSTIC TOOL")
    print("=" * 100)
    print()

    # Find .emd files in data directory
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent.parent / "results"

    datasets_to_test = []

    # Map .emd files to their metadata
    emd_files = list(data_dir.glob("*.emd"))

    for emd_file in emd_files:
        # Match to metadata
        dataset_name = emd_file.stem

        # Find corresponding metadata
        metadata_file = results_dir / dataset_name / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
        else:
            meta = {}

        # Categorize by type
        if "4D_EELS" in dataset_name or "3D_EELS" in dataset_name:
            dtype = "4D EELS"
        elif "4D_Diff" in dataset_name:
            dtype = "4D Diffraction"
        else:
            dtype = "Unknown"

        datasets_to_test.append(
            {"name": dataset_name, "path": emd_file, "type": dtype, "metadata": meta}
        )

    if len(datasets_to_test) == 0:
        print("⚠ No datasets found to test!")
        print("\nSearched in:", data_dir)
        return

    print(f"Found {len(datasets_to_test)} datasets to test\n")

    # Run diagnostics on each dataset
    all_results = {}

    for dataset_info in datasets_to_test:
        name = dataset_info["name"]
        filepath = dataset_info["path"]
        dtype = dataset_info["type"]

        print("=" * 100)
        print(f"TESTING: {name} ({dtype})")
        print("=" * 100)

        # Check if file exists
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            continue

        print(f"  File: {filepath}")
        print()

        # Test 1: Storage device
        print("1. STORAGE DEVICE CHECK")
        print("-" * 100)
        device_info = get_file_device(filepath)
        if device_info:
            for key, value in device_info.items():
                print(f"  {key}: {value}")
        print()

        # Test 2: File information
        print("2. FILE INFORMATION")
        print("-" * 100)
        file_info = get_file_info(filepath)
        for key, value in file_info.items():
            print(f"  {key}: {value}")
        print()

        # Test 3: HDF5 structure
        print("3. HDF5 STRUCTURE")
        print("-" * 100)
        try:
            with h5py.File(filepath, "r") as f:
                # Find the data array
                def find_data(name, obj):
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) == 4:
                        print(f"  Dataset: {name}")
                        print(f"    Shape: {obj.shape}")
                        print(f"    Dtype: {obj.dtype}")
                        print(f"    Chunks: {obj.chunks}")
                        print(f"    Compression: {obj.compression}")
                        print(f"    Size: {obj.nbytes / (1024**3):.2f} GiB")

                f.visititems(find_data)
        except Exception as e:
            print(f"  ✗ Error reading HDF5: {e}")
        print()

        # Test 4: Read speed tests
        print("4. READ SPEED TESTS")
        print("-" * 100)

        # Find the dataset path
        dataset_path = None
        with h5py.File(filepath, "r") as f:

            def find_path(name, obj):
                nonlocal dataset_path
                if (
                    isinstance(obj, h5py.Dataset)
                    and len(obj.shape) == 4
                    and dataset_path is None
                ):
                    dataset_path = name

            f.visititems(find_path)

        if dataset_path is None:
            print("  ✗ Could not find 4D dataset in file")
            continue

        print(f"  Dataset path: {dataset_path}")
        print()

        # Test cold cache (if possible)
        print("  Test A: Cold cache read")
        cold_results = test_read_speed(
            filepath, dataset_path, "Cold", clear_cache_first=True
        )
        print(f"    Open time: {cold_results['open_time']:.4f} s")
        print(f"    Metadata time: {cold_results['metadata_time']:.4f} s")
        print(f"    Read time: {cold_results['read_time']:.4f} s")
        print(f"    Throughput: {cold_results['throughput_gbs']:.2f} GiB/s")
        print()

        # Test warm cache
        print("  Test B: Warm cache read (immediate re-read)")
        print(f"    Read time: {cold_results['read_time_cached']:.4f} s")
        print(f"    Throughput: {cold_results['throughput_gbs_cached']:.2f} GiB/s")
        print(
            f"    Speedup: {cold_results['throughput_gbs_cached'] / cold_results['throughput_gbs']:.1f}×"
        )
        print()

        # Test partial reads
        print("  Test C: Partial read patterns")
        partial_results = test_partial_reads(filepath, dataset_path)
        print(
            f"    Single frame: {partial_results['single_frame_mean_time'] * 1000:.2f} ± {partial_results['single_frame_std_time'] * 1000:.2f} ms"
        )
        print(
            f"    Slice (10 frames): {partial_results['slice_mean_time'] * 1000:.2f} ± {partial_results['slice_std_time'] * 1000:.2f} ms"
        )
        print()

        # Store results
        all_results[name] = {
            "device": device_info,
            "file_info": file_info,
            "cold_cache": cold_results,
            "partial_reads": partial_results,
        }

    # Summary comparison
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)

    if len(all_results) >= 2:
        names = list(all_results.keys())

        print(f"\n{'Metric':<40} {names[0]:<25} {names[1]:<25} {'Ratio':<10}")
        print("-" * 100)

        for name in names:
            r = all_results[name]
            print(f"\nDataset: {name}")
            print(f"  Device: {r['device'].get('device', 'N/A')}")
            print(f"  Mountpoint: {r['device'].get('mountpoint', 'N/A')}")
            print(f"  File size: {r['file_info']['size_gb']:.2f} GiB")
            print(
                f"  Cold cache throughput: {r['cold_cache']['throughput_gbs']:.2f} GiB/s"
            )
            print(
                f"  Warm cache throughput: {r['cold_cache']['throughput_gbs_cached']:.2f} GiB/s"
            )
            print(
                f"  Cache speedup: {r['cold_cache']['throughput_gbs_cached'] / r['cold_cache']['throughput_gbs']:.1f}×"
            )

        # Calculate ratios
        if len(names) == 2:
            r1 = all_results[names[0]]
            r2 = all_results[names[1]]

            print(f"\nSpeed ratio ({names[1]} / {names[0]}):")
            cold_ratio = (
                r2["cold_cache"]["throughput_gbs"] / r1["cold_cache"]["throughput_gbs"]
            )
            warm_ratio = (
                r2["cold_cache"]["throughput_gbs_cached"]
                / r1["cold_cache"]["throughput_gbs_cached"]
            )
            print(f"  Cold cache: {cold_ratio:.1f}×")
            print(f"  Warm cache: {warm_ratio:.1f}×")

            if r1["device"]["device"] == r2["device"]["device"]:
                print(f"\n✓ Both datasets on same device: {r1['device']['device']}")
            else:
                print(f"\n⚠ Datasets on DIFFERENT devices!")
                print(f"  {names[0]}: {r1['device']['device']}")
                print(f"  {names[1]}: {r2['device']['device']}")

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
