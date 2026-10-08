#!/usr/bin/env python3
"""
Command-line script to run compression benchmarks on 4D STEM datasets.

Usage:
    python run_benchmark.py <dataset.emd> [--output DIR] [--name NAME]

Examples:
    python run_benchmark.py /path/to/dataset.emd
    python run_benchmark.py /path/to/dataset.emd --name my_dataset
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
  %(prog)s /path/to/dataset.emd
  %(prog)s /path/to/dataset.emd --name my_dataset
        """,
    )
    parser.add_argument("dataset", type=str, help="Path to EMD/HDF5 dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        # Anchored on this file: "../results" meant the repository's results/
        # from implementation/, but implementation/results/ from src/.
        default=str(Path(__file__).resolve().parents[2] / "results"),
        help="Output directory for results (default: the repository's results/)",
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
    print("  1. Repeat for the other datasets, or run run_multiple_benchmarks.py")
    print("  2. Aggregate: uv run python implementation/src/aggregate_multi_run_results.py")
    print("  3. Rebuild the artifacts: cd implementation/src &&"
          " uv run python -m paper_artifacts.generate_all")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
