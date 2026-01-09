#!/usr/bin/env python3
"""
Run Benchmarks Multiple Times for Statistical Analysis

This script runs the full benchmark suite N times, storing each run in a separate
timestamped directory. This allows for statistical analysis of performance variability.

Usage:
    python run_multiple_benchmarks.py --n-runs 10
    python run_multiple_benchmarks.py --n-runs 5 --start-run 3  # Resume from run 3
    python run_multiple_benchmarks.py --n-runs 10 --datasets 4D_EELS 3D_EELS  # Specific datasets only

Directory structure created:
    results/
    ├── run_001_20241205_143022/
    │   ├── 4D_EELS/
    │   ├── 4D_Diff/
    │   └── ...
    ├── run_002_20241205_150145/
    │   └── ...
    └── run_010_20241205_183456/
        └── ...

After completion, use aggregate_multi_run_results.py to compute statistics.
"""

import argparse
from pathlib import Path
import subprocess
import time
import sys
from datetime import datetime
import json


def get_timestamp():
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_directory(base_dir, run_number):
    """
    Create a timestamped directory for this run.
    
    Parameters
    ----------
    base_dir : Path
        Base results directory
    run_number : int
        Run number (1-indexed)
    
    Returns
    -------
    Path
        Created directory path
    """
    timestamp = get_timestamp()
    run_dir = base_dir / f"run_{run_number:03d}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir, run_number, total_runs, start_time, args):
    """Save metadata about this run."""
    metadata = {
        'run_number': run_number,
        'total_runs': total_runs,
        'start_time': start_time.isoformat(),
        'timestamp': get_timestamp(),
        'data_dir': str(args.data_dir),
        'datasets': args.datasets if args.datasets else 'all',
        'skip_datasets': args.skip if args.skip else [],
        'command': ' '.join(sys.argv)
    }
    
    metadata_file = run_dir / 'run_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def run_single_benchmark(run_dir, data_dir, datasets=None, skip=None, verbose=True):
    """
    Run a single benchmark iteration.
    
    Parameters
    ----------
    run_dir : Path
        Output directory for this run
    data_dir : Path
        Directory containing .emd files
    datasets : list of str, optional
        Specific datasets to run (if None, runs all)
    skip : list of str, optional
        Datasets to skip
    verbose : bool
        Print detailed output
    
    Returns
    -------
    int
        Return code (0 = success)
    """
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        'run_all_benchmarks.py',
        '--data-dir', str(data_dir),
        '--output-dir', str(run_dir),
        '--yes'  # Skip confirmation prompt
    ]
    
    # Add dataset filters
    if datasets:
        for dataset in datasets:
            cmd.extend(['--only', dataset])
    
    if skip:
        for dataset in skip:
            cmd.extend(['--skip', dataset])
    
    if not verbose:
        cmd.append('--quiet')
    
    # Run benchmark
    script_dir = Path(__file__).parent
    result = subprocess.run(cmd, cwd=script_dir)
    
    return result.returncode


def print_progress_summary(run_number, total_runs, run_dir, elapsed_time, success):
    """Print summary after each run."""
    print("\n" + "="*100)
    print(f"RUN {run_number}/{total_runs} {'COMPLETED' if success else 'FAILED'}")
    print("="*100)
    print(f"Output directory: {run_dir}")
    print(f"Elapsed time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    
    if run_number < total_runs:
        estimated_remaining = elapsed_time * (total_runs - run_number)
        print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes ({estimated_remaining/3600:.2f} hours)")
    
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run compression benchmarks multiple times for statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 complete benchmark iterations
  python run_multiple_benchmarks.py --n-runs 10
  
  # Run 5 iterations on specific datasets only
  python run_multiple_benchmarks.py --n-runs 5 --datasets 4D_EELS 3D_EELS
  
  # Resume from run 6 (if previous runs were interrupted)
  python run_multiple_benchmarks.py --n-runs 10 --start-run 6
  
  # Run with custom data directory
  python run_multiple_benchmarks.py --n-runs 10 --data-dir /path/to/data --output-base /path/to/results

After completion:
  python aggregate_multi_run_results.py --results-dir ../../results
        """
    )
    
    parser.add_argument('--n-runs', type=int, default=10,
                       help='Number of complete benchmark runs (default: 10)')
    parser.add_argument('--start-run', type=int, default=1,
                       help='Starting run number (for resuming, default: 1)')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing .emd files (default: ../data)')
    parser.add_argument('--output-base', type=str, default='../../results',
                       help='Base directory for results (default: ../../results)')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to run (default: all)')
    parser.add_argument('--skip', nargs='+', default=None,
                       help='Datasets to skip')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress (default: True)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--pause', type=int, default=5,
                       help='Seconds to pause between runs (default: 5)')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop if any run fails (default: continue)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_base = Path(args.output_base)
    
    if args.quiet:
        args.verbose = False
    
    # Validate inputs
    if args.start_run < 1 or args.start_run > args.n_runs:
        print(f"Error: --start-run must be between 1 and {args.n_runs}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*100)
    print("MULTIPLE BENCHMARK RUNS FOR STATISTICAL ANALYSIS")
    print("="*100)
    print(f"Total runs: {args.n_runs}")
    print(f"Starting from run: {args.start_run}")
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Output base: {output_base.resolve()}")
    
    if args.datasets:
        print(f"Datasets: {', '.join(args.datasets)}")
    else:
        print("Datasets: All .emd files in data directory")
    
    if args.skip:
        print(f"Skipping: {', '.join(args.skip)}")
    
    print("="*100 + "\n")
    
    # Track overall progress
    overall_start = time.time()
    run_results = []
    
    # Run benchmarks
    for run_num in range(args.start_run, args.n_runs + 1):
        print(f"\n{'='*100}")
        print(f"STARTING RUN {run_num}/{args.n_runs}")
        print(f"{'='*100}\n")
        
        # Create run directory
        run_dir = create_run_directory(output_base, run_num)
        print(f"Output directory: {run_dir}\n")
        
        # Save metadata
        run_start = datetime.now()
        save_run_metadata(run_dir, run_num, args.n_runs, run_start, args)
        
        # Run benchmark
        run_start_time = time.time()
        
        try:
            returncode = run_single_benchmark(
                run_dir=run_dir,
                data_dir=data_dir,
                datasets=args.datasets,
                skip=args.skip,
                verbose=args.verbose
            )
            
            success = (returncode == 0)
            
        except Exception as e:
            print(f"\n✗ Run {run_num} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            success = False
            returncode = 1
        
        run_elapsed = time.time() - run_start_time
        
        # Record result
        run_results.append({
            'run_number': run_num,
            'success': success,
            'elapsed_time': run_elapsed,
            'output_dir': str(run_dir)
        })
        
        # Print progress
        print_progress_summary(run_num, args.n_runs, run_dir, run_elapsed, success)
        
        # Handle failures
        if not success and args.stop_on_error:
            print(f"\n✗ Stopping due to failure in run {run_num} (--stop-on-error)")
            break
        
        # Pause between runs (except after last run)
        if run_num < args.n_runs and args.pause > 0:
            print(f"Pausing {args.pause} seconds before next run...\n")
            time.sleep(args.pause)
    
    # Print final summary
    total_elapsed = time.time() - overall_start
    successful_runs = sum(1 for r in run_results if r['success'])
    failed_runs = len(run_results) - successful_runs
    
    print("\n" + "="*100)
    print("MULTI-RUN BENCHMARK COMPLETE")
    print("="*100)
    print(f"Total runs attempted: {len(run_results)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Average time per run: {total_elapsed/len(run_results)/60:.1f} minutes")
    print()
    
    # Print individual run summary
    print(f"{'Run':<8} {'Status':<12} {'Time (min)':<12} {'Output Directory'}")
    print("-"*100)
    for r in run_results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"{r['run_number']:<8} {status:<12} {r['elapsed_time']/60:>10.1f}  {r['output_dir']}")
    
    print("="*100)
    
    # Save summary
    summary_file = output_base / f"multi_run_summary_{get_timestamp()}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_runs': len(run_results),
            'successful': successful_runs,
            'failed': failed_runs,
            'total_time_hours': total_elapsed / 3600,
            'runs': run_results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Next steps
    if successful_runs > 0:
        print("\n" + "="*100)
        print("NEXT STEPS")
        print("="*100)
        print("1. Aggregate results and compute statistics:")
        print(f"   python aggregate_multi_run_results.py --results-dir {output_base}")
        print()
        print("2. Generate figures with error bars:")
        print("   python generate_all_figures.py --use-statistics")
        print()
        print("3. Update paper with statistical analysis")
        print("="*100)
    
    # Exit code
    if failed_runs > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
