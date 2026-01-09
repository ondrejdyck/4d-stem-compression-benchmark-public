#!/usr/bin/env python3
"""
Command-line script to run compression benchmarks on 4D STEM datasets.

Usage:
    python run_benchmark.py <dataset.emd> [--output results/] [--name dataset_name]

Examples:
    python run_benchmark.py ../data/4D_EELS.emd
    python run_benchmark.py ../data/4D_Diff.emd --name 4D_Diff_test
"""

import argparse
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from compression_benchmark import load_emd, run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run 4D STEM compression benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ../data/4D_EELS.emd
  %(prog)s ../data/4D_Diff.emd --output ../results --name 4D_Diff_test
        """,
    )
    parser.add_argument("dataset", type=str, help="Path to EMD/HDF5 dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../results",
        help="Output directory for results (default: ../results)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Dataset name (default: filename without extension)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Create summary plots (default: False, do in notebook)",
    )

    args = parser.parse_args()

    # Setup paths
    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f"ERROR: Dataset not found: {dataset_file}")
        return 1

    # Determine dataset name
    dataset_name = args.name if args.name else dataset_file.stem.replace(" ", "_")
    output_dir = Path(args.output) / dataset_name

    print("=" * 70)
    print("4D STEM COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Dataset: {dataset_file}")
    print(f"Output:  {output_dir}")
    print(f"Name:    {dataset_name}")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading {dataset_file.name}...")
    start_time = time.time()
    try:
        data_4d = load_emd(dataset_file)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return 1

    load_time = time.time() - start_time
    print(f"✓ Loaded in {load_time:.1f}s")
    print(f"  Shape: {data_4d.shape}")
    print(f"  Size: {data_4d.nbytes / (1024**3):.2f} GiB")
    print()

    # Run benchmark
    try:
        results = run_benchmark(
            data_4d, output_dir, dataset_name, save_csv=True, create_plots=args.plots
        )
    except Exception as e:
        print(f"ERROR during benchmark: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"✓ Results saved to: {output_dir}")
    print(f"  - CSV file: benchmark_results.csv")
    print(f"  - Metadata: metadata.json")
    print(f"  - Details: {dataset_name}_detailed_results.txt")
    if args.plots:
        print(f"  - Plot: compression_benchmark.png")
    print()
    print("Next steps:")
    print("  1. Open notebooks/visualize_results.ipynb")
    print("  2. Load the CSV file")
    print("  3. Create custom plots")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
