#!/usr/bin/env python3
"""Quick test of fixed benchmark"""

import numpy as np
from pathlib import Path
from compression_benchmark import run_benchmark

# Create small test data
print("Creating small test dataset...")
test_data = np.random.randint(0, 100, size=(32, 32, 64, 64), dtype="uint16")
print(f"Test data shape: {test_data.shape}")
print(f"Test data size: {test_data.nbytes / (1024**2):.1f} MiB")

# Run benchmark with just a few methods
output_dir = Path("../results/test_fixed_benchmark")
output_dir.mkdir(parents=True, exist_ok=True)

print("\nRunning benchmark...")
results = run_benchmark(test_data, output_dir=output_dir, dataset_name="test_fixed")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for method_name, method_data in results["compression_results"].items():
    if "balanced" in method_name:  # Only show balanced chunking
        print(f"\n{method_name}:")
        print(f"  Compression: {method_data['compression_ratio']:.1f}×")
        print(
            f"  Write: {method_data['write_time']:.3f}s ({method_data.get('write_throughput_gbs', 0):.2f} GiB/s)"
        )
        print(
            f"  Read: {method_data['read_time']:.3f}s ({method_data.get('read_throughput_gbs', 0):.2f} GiB/s)"
        )
        if "bytes_read" in method_data:
            print(f"  Bytes read: {method_data['bytes_read'] / (1024**2):.1f} MiB")
        if "actual_io_bytes" in method_data:
            print(f"  Actual I/O: {method_data['actual_io_bytes'] / (1024**2):.1f} MiB")

print("\n✓ Test complete!")
