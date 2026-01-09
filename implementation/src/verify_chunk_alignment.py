#!/usr/bin/env python3
"""
Chunk Alignment Verification Test

Tests the hypothesis that the 13× read speed difference is caused by poor chunk
alignment between the data shape and chunk size.

Hypothesis:
- EELS (64, 64, 256, 1024) with chunks (16, 16, 128, 128) requires reading many
  more chunks because 1024/128 = 8 chunks in detector X dimension
- Diffraction (256, 256, 256, 256) with chunks (16, 16, 128, 128) requires fewer
  chunks because 256/128 = 2 chunks in each dimension

This test will:
1. Create test HDF5 files with the same chunking as the benchmark
2. Measure read times for the exact same slices the benchmark uses
3. Count how many chunks are actually accessed
4. Verify if chunk count correlates with read time
"""

import h5py
import numpy as np
import time
from pathlib import Path
import tempfile


def create_test_file(shape, chunks, dtype="uint16", compression=None):
    """Create a test HDF5 file with specified shape and chunking."""
    # Create temporary file
    fd, filepath = tempfile.mkstemp(suffix=".h5")
    filepath = Path(filepath)

    # Create random data
    data = np.random.randint(0, 100, size=shape, dtype=dtype)

    # Write to HDF5
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=data, chunks=chunks, compression=compression)

    return filepath


def calculate_chunks_accessed(data_shape, chunk_shape, slice_tuple):
    """
    Calculate how many chunks need to be accessed for a given slice.

    Parameters
    ----------
    data_shape : tuple
        Shape of the dataset
    chunk_shape : tuple
        Shape of chunks
    slice_tuple : tuple of slices
        The slice being accessed

    Returns
    -------
    int
        Number of chunks that need to be read
    """
    chunks_per_dim = []

    for dim_idx, (dim_size, chunk_size, slice_obj) in enumerate(
        zip(data_shape, chunk_shape, slice_tuple)
    ):
        if isinstance(slice_obj, slice):
            start = slice_obj.start if slice_obj.start is not None else 0
            stop = slice_obj.stop if slice_obj.stop is not None else dim_size
        elif isinstance(slice_obj, int):
            start = slice_obj
            stop = slice_obj + 1
        else:
            start = 0
            stop = dim_size

        # Calculate which chunks are touched
        first_chunk = start // chunk_size
        last_chunk = (stop - 1) // chunk_size
        num_chunks = last_chunk - first_chunk + 1
        chunks_per_dim.append(num_chunks)

    # Total chunks is product of chunks in each dimension
    total_chunks = np.prod(chunks_per_dim)

    return total_chunks, chunks_per_dim


def test_read_pattern(filepath, read_slices, description):
    """
    Test reading with specific slice patterns and measure time.

    Parameters
    ----------
    filepath : Path
        Path to HDF5 file
    read_slices : list of tuples
        List of slice tuples to read
    description : str
        Description of this test

    Returns
    -------
    dict
        Results including timing and chunk counts
    """
    results = {
        "description": description,
        "slices": [],
        "total_time": 0,
        "total_chunks": 0,
        "total_bytes": 0,
    }

    with h5py.File(filepath, "r") as f:
        dset = f["data"]
        data_shape = dset.shape
        chunk_shape = dset.chunks
        dtype_size = dset.dtype.itemsize

        # Warm up (open file, access metadata)
        _ = dset.shape

        # Test each slice
        for slice_tuple in read_slices:
            # Calculate expected chunks
            num_chunks, chunks_per_dim = calculate_chunks_accessed(
                data_shape, chunk_shape, slice_tuple
            )

            # Measure read time
            start = time.time()
            data = dset[slice_tuple]
            elapsed = time.time() - start

            # Calculate bytes read
            bytes_read = data.nbytes

            results["slices"].append(
                {
                    "slice": str(slice_tuple),
                    "time": elapsed,
                    "chunks_accessed": num_chunks,
                    "chunks_per_dim": chunks_per_dim,
                    "bytes_read": bytes_read,
                    "throughput_mbs": bytes_read / elapsed / (1024**2),
                }
            )

            results["total_time"] += elapsed
            results["total_chunks"] += num_chunks
            results["total_bytes"] += bytes_read

    results["avg_throughput_mbs"] = (
        results["total_bytes"] / results["total_time"] / (1024**2)
    )

    return results


def main():
    print("=" * 100)
    print("CHUNK ALIGNMENT VERIFICATION TEST")
    print("=" * 100)
    print()

    # Define test configurations matching the benchmark
    configs = [
        {
            "name": "64x64 EELS",
            "shape": (64, 64, 256, 1024),
            "chunks": (16, 16, 128, 128),
            "dtype": "uint16",
        },
        {
            "name": "256x256 Diffraction",
            "shape": (256, 256, 256, 256),
            "chunks": (16, 16, 128, 128),
            "dtype": "uint16",
        },
    ]

    # Define read patterns matching the benchmark (lines 228-230)
    def get_read_slices(sy, sx, qy, qx):
        """Generate the same read slices as the benchmark."""
        read_sy = min(32, sy)
        read_sx = min(32, sx)
        read_qy = min(100, qy - 1)
        read_qx = min(100, qx - 1)

        return [
            (
                slice(0, read_sy),
                slice(0, read_sx),
                slice(None),
                slice(None),
            ),  # One chunk
            (0, 0, slice(None), slice(None)),  # Single frame
            (
                slice(0, min(64, sy)),
                slice(0, min(64, sx)),
                read_qy,
                read_qx,
            ),  # K-space slice
        ]

    all_results = {}

    for config in configs:
        print(f"\n{'=' * 100}")
        print(f"TESTING: {config['name']}")
        print(f"{'=' * 100}")
        print(f"Shape: {config['shape']}")
        print(f"Chunks: {config['chunks']}")
        print(f"Dtype: {config['dtype']}")
        print()

        # Create test file
        print("Creating test file...")
        filepath = create_test_file(config["shape"], config["chunks"], config["dtype"])
        file_size_mb = filepath.stat().st_size / (1024**2)
        print(f"✓ Created: {filepath}")
        print(f"  File size: {file_size_mb:.1f} MiB")
        print()

        # Get read slices
        sy, sx, qy, qx = config["shape"]
        read_slices = get_read_slices(sy, sx, qy, qx)

        # Test reads
        print("Testing read patterns...")
        results = test_read_pattern(filepath, read_slices, config["name"])

        # Print detailed results
        print(f"\nDetailed Results:")
        print(
            f"{'Slice':<50} {'Time (ms)':<12} {'Chunks':<10} {'Chunks/Dim':<20} {'MiB Read':<10} {'MiB/s':<10}"
        )
        print("-" * 100)

        for i, slice_result in enumerate(results["slices"], 1):
            print(
                f"{slice_result['slice']:<50} "
                f"{slice_result['time'] * 1000:>10.2f}  "
                f"{slice_result['chunks_accessed']:>8}  "
                f"{str(slice_result['chunks_per_dim']):<20} "
                f"{slice_result['bytes_read'] / (1024**2):>8.1f}  "
                f"{slice_result['throughput_mbs']:>8.1f}"
            )

        print("-" * 100)
        print(
            f"{'TOTAL':<50} "
            f"{results['total_time'] * 1000:>10.2f}  "
            f"{results['total_chunks']:>8}  "
            f"{'':20} "
            f"{results['total_bytes'] / (1024**2):>8.1f}  "
            f"{results['avg_throughput_mbs']:>8.1f}"
        )

        # Store results
        all_results[config["name"]] = results

        # Clean up
        filepath.unlink()
        print(f"\n✓ Cleaned up test file")

    # Comparison
    print("\n" + "=" * 100)
    print("COMPARISON & VERIFICATION")
    print("=" * 100)

    if len(all_results) >= 2:
        names = list(all_results.keys())
        r1 = all_results[names[0]]
        r2 = all_results[names[1]]

        print(f"\n{names[0]} vs {names[1]}:")
        print("-" * 100)
        print(
            f"Total chunks accessed: {r1['total_chunks']} vs {r2['total_chunks']} "
            f"(ratio: {r1['total_chunks'] / r2['total_chunks']:.1f}×)"
        )
        print(
            f"Total bytes read: {r1['total_bytes'] / (1024**2):.1f} MiB vs {r2['total_bytes'] / (1024**2):.1f} MiB "
            f"(ratio: {r1['total_bytes'] / r2['total_bytes']:.1f}×)"
        )
        print(
            f"Total read time: {r1['total_time'] * 1000:.1f} ms vs {r2['total_time'] * 1000:.1f} ms "
            f"(ratio: {r1['total_time'] / r2['total_time']:.1f}×)"
        )
        print(
            f"Average throughput: {r1['avg_throughput_mbs']:.1f} MiB/s vs {r2['avg_throughput_mbs']:.1f} MiB/s "
            f"(ratio: {r2['avg_throughput_mbs'] / r1['avg_throughput_mbs']:.1f}×)"
        )

        print("\n" + "=" * 100)
        print("HYPOTHESIS VERIFICATION")
        print("=" * 100)

        chunk_ratio = r1["total_chunks"] / r2["total_chunks"]
        time_ratio = r1["total_time"] / r2["total_time"]

        print(f"\nChunk count ratio: {chunk_ratio:.1f}×")
        print(f"Read time ratio: {time_ratio:.1f}×")

        if abs(chunk_ratio - time_ratio) / chunk_ratio < 0.3:  # Within 30%
            print(f"\n✓ HYPOTHESIS CONFIRMED!")
            print(
                f"  The read time ratio ({time_ratio:.1f}×) closely matches the chunk count ratio ({chunk_ratio:.1f}×)"
            )
            print(
                f"  This confirms that poor chunk alignment is the primary cause of the speed difference."
            )
        else:
            print(f"\n✗ HYPOTHESIS REJECTED")
            print(
                f"  The read time ratio ({time_ratio:.1f}×) does NOT match the chunk count ratio ({chunk_ratio:.1f}×)"
            )
            print(f"  Other factors besides chunk alignment must be involved.")

        # Detailed slice-by-slice analysis
        print("\n" + "=" * 100)
        print("SLICE-BY-SLICE ANALYSIS")
        print("=" * 100)

        for i in range(len(r1["slices"])):
            s1 = r1["slices"][i]
            s2 = r2["slices"][i]

            print(f"\nSlice {i + 1}: {s1['slice']}")
            print(
                f"  {names[0]}: {s1['chunks_accessed']} chunks, {s1['time'] * 1000:.2f} ms, {s1['throughput_mbs']:.1f} MiB/s"
            )
            print(
                f"  {names[1]}: {s2['chunks_accessed']} chunks, {s2['time'] * 1000:.2f} ms, {s2['throughput_mbs']:.1f} MiB/s"
            )
            print(
                f"  Chunk ratio: {s1['chunks_accessed'] / s2['chunks_accessed']:.1f}×"
            )
            print(f"  Time ratio: {s1['time'] / s2['time']:.1f}×")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
