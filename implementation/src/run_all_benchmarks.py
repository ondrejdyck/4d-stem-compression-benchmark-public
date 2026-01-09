#!/usr/bin/env python3
"""
Run Benchmarks on All Datasets

Automatically discovers and benchmarks all .emd files in the data directory.
Uses the fixed benchmark code with proper throughput measurement.

Usage:
    python run_all_benchmarks.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--skip DATASET_NAME]
"""

import argparse
from pathlib import Path
import time
import sys
from compression_benchmark import load_emd, run_benchmark, save_results_to_csv


def find_datasets(data_dir):
    """
    Find all .emd files in the data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing .emd files

    Returns
    -------
    list of dict
        List of dataset info dicts with 'name' and 'path' keys
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    datasets = []
    for emd_file in sorted(data_dir.glob("*.emd")):
        datasets.append(
            {
                "name": emd_file.stem,
                "path": emd_file,
                "size_mb": emd_file.stat().st_size / (1024**2),
            }
        )

    return datasets


def run_benchmark_on_dataset(dataset_info, output_base_dir, verbose=True):
    """
    Run benchmark on a single dataset.

    Parameters
    ----------
    dataset_info : dict
        Dataset information with 'name', 'path', 'size_mb'
    output_base_dir : Path
        Base directory for output (will create subdirectory per dataset)
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Benchmark results
    """
    name = dataset_info["name"]
    filepath = dataset_info["path"]

    if verbose:
        print("\n" + "=" * 100)
        print(f"BENCHMARKING: {name}")
        print("=" * 100)
        print(f"File: {filepath}")
        print(f"Size: {dataset_info['size_mb']:.1f} MiB")
        print()

    # Load data
    try:
        if verbose:
            print("Loading data...")
        start_load = time.time()
        data = load_emd(filepath)
        load_time = time.time() - start_load

        if verbose:
            print(f"✓ Loaded in {load_time:.1f}s")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Size: {data.nbytes / (1024**3):.2f} GiB")
            print()
    except Exception as e:
        print(f"✗ Failed to load {name}: {e}")
        return None

    # Create output directory
    output_dir = output_base_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    try:
        if verbose:
            print("Running benchmark...")
        start_bench = time.time()

        results = run_benchmark(data, output_dir=output_dir, dataset_name=name)

        bench_time = time.time() - start_bench

        if verbose:
            print(f"\n✓ Benchmark complete in {bench_time / 60:.1f} minutes")
            print(f"  Results saved to: {output_dir}")

        return results

    except Exception as e:
        print(f"✗ Benchmark failed for {name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run compression benchmarks on all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all datasets in default location
  python run_all_benchmarks.py
  
  # Specify custom directories
  python run_all_benchmarks.py --data-dir /path/to/data --output-dir /path/to/results
  
  # Skip specific datasets
  python run_all_benchmarks.py --skip "4D_EELS" --skip "4D_Diff"
  
  # Dry run (list datasets without running)
  python run_all_benchmarks.py --dry-run
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing .emd files (default: ../data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../results",
        help="Base directory for results (default: ../../results)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        action="append",
        default=[],
        help="Dataset names to skip (can be used multiple times)",
    )
    parser.add_argument(
        "--only",
        type=str,
        action="append",
        default=[],
        help="Only run these datasets (can be used multiple times)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List datasets without running benchmarks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.quiet:
        args.verbose = False

    # Find datasets
    print("=" * 100)
    print("BENCHMARK ALL DATASETS")
    print("=" * 100)
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    try:
        datasets = find_datasets(data_dir)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    if len(datasets) == 0:
        print(f"✗ No .emd files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(datasets)} datasets:")
    print()

    # Filter datasets
    if args.only:
        datasets = [d for d in datasets if d["name"] in args.only]
        print(f"Filtered to {len(datasets)} datasets (--only)")

    if args.skip:
        datasets = [d for d in datasets if d["name"] not in args.skip]
        print(f"Filtered to {len(datasets)} datasets (--skip)")

    # Display datasets
    total_size_gb = sum(d["size_mb"] for d in datasets) / 1024

    for i, dataset in enumerate(datasets, 1):
        status = "SKIP" if dataset["name"] in args.skip else "RUN"
        print(
            f"  {i}. {dataset['name']:<40} {dataset['size_mb']:>8.1f} MiB  [{status}]"
        )

    print()
    print(f"Total data size: {total_size_gb:.2f} GiB")
    print()

    if args.dry_run:
        print("✓ Dry run complete (no benchmarks executed)")
        sys.exit(0)

    # Confirm before running
    if not args.quiet and not args.yes:
        print("=" * 100)
        try:
            response = input(f"Run benchmarks on {len(datasets)} datasets? [y/N]: ")
            if response.lower() not in ["y", "yes"]:
                print("Cancelled.")
                sys.exit(0)
        except EOFError:
            print("\nNo input available. Use --yes to skip confirmation.")
            sys.exit(1)

    # Run benchmarks
    print("\n" + "=" * 100)
    print("STARTING BENCHMARKS")
    print("=" * 100)

    start_time = time.time()
    results_summary = []

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing: {dataset['name']}")

        dataset_start = time.time()
        results = run_benchmark_on_dataset(dataset, output_dir, verbose=args.verbose)
        dataset_time = time.time() - dataset_start

        results_summary.append(
            {
                "name": dataset["name"],
                "success": results is not None,
                "time_minutes": dataset_time / 60,
                "size_mb": dataset["size_mb"],
            }
        )

        if results is not None:
            print(f"✓ {dataset['name']} completed in {dataset_time / 60:.1f} minutes")
        else:
            print(f"✗ {dataset['name']} FAILED")

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print()

    successful = sum(1 for r in results_summary if r["success"])
    failed = len(results_summary) - successful

    print(f"Total datasets: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    print()

    print(f"{'Dataset':<40} {'Size (MiB)':<12} {'Time (min)':<12} {'Status':<10}")
    print("-" * 100)

    for r in results_summary:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        time_str = f"{r['time_minutes']:.1f}" if r["success"] else "N/A"
        print(f"{r['name']:<40} {r['size_mb']:>10.1f}  {time_str:>10}  {status}")

    print("=" * 100)

    if failed > 0:
        print(f"\n⚠ {failed} dataset(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print(f"\n✓ All benchmarks completed successfully!")
        print(f"\nResults saved to: {output_dir.resolve()}")
        print("\nNext steps:")
        print("  1. Review results in output directory")
        print("  2. Generate visualizations: python generate_all_figures.py")
        print("  3. Update paper with new results")
        sys.exit(0)


if __name__ == "__main__":
    main()
