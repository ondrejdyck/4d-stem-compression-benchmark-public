#!/usr/bin/env python3
"""
Aggregate Multi-Run Benchmark Results

This script combines results from multiple benchmark runs and computes statistics
(mean, std, min, max, median, CV%) for each method across all runs.

Usage:
    python aggregate_multi_run_results.py
    python aggregate_multi_run_results.py --results-dir ../../results
    python aggregate_multi_run_results.py --output aggregated_stats.csv

Input structure:
    results/
    ├── run_001_20241205_143022/
    │   ├── 4D_EELS/benchmark_results.csv
    │   ├── 4D_Diff/benchmark_results.csv
    │   └── ...
    ├── run_002_20241205_150145/
    │   └── ...
    └── run_010_20241205_183456/
        └── ...

Output:
    results/aggregated/
    ├── statistics.csv              # Mean, std, min, max for all metrics
    ├── all_runs_combined.csv       # All raw data with run_number column
    ├── metadata.json               # Info about aggregation
    └── summary_report.txt          # Human-readable summary
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime


def find_run_directories(results_dir):
    """
    Find all run_XXX directories in results folder.
    
    Parameters
    ----------
    results_dir : Path
        Base results directory
    
    Returns
    -------
    list of Path
        Sorted list of run directories
    """
    results_dir = Path(results_dir)
    
    # Find directories matching run_XXX pattern
    run_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith('run_'):
            run_dirs.append(item)
    
    # Sort by run number
    run_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    return run_dirs


def load_all_runs(run_dirs, verbose=True):
    """
    Load benchmark results from all runs.
    
    Parameters
    ----------
    run_dirs : list of Path
        List of run directories
    verbose : bool
        Print progress
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all runs
    """
    all_data = []
    
    for run_dir in run_dirs:
        # Extract run number from directory name
        run_num = int(run_dir.name.split('_')[1])
        
        if verbose:
            print(f"Loading run {run_num:03d}: {run_dir.name}")
        
        # Find all dataset subdirectories
        dataset_count = 0
        for dataset_dir in run_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            csv_file = dataset_dir / 'benchmark_results.csv'
            if not csv_file.exists():
                continue
            
            # Load CSV
            df = pd.read_csv(csv_file)
            df['run_number'] = run_num
            df['run_directory'] = run_dir.name
            
            all_data.append(df)
            dataset_count += 1
        
        if verbose:
            print(f"  Loaded {dataset_count} datasets")
    
    if not all_data:
        raise ValueError("No benchmark results found in run directories")
    
    # Combine all runs
    combined = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f"\nTotal records loaded: {len(combined)}")
        print(f"Runs: {combined['run_number'].nunique()}")
        print(f"Datasets: {combined['dataset'].nunique()}")
        print(f"Methods: {combined['method'].nunique()}")
    
    return combined


def compute_statistics(combined_df, group_by=['dataset', 'method']):
    """
    Compute statistics for each dataset/method combination.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined data from all runs
    group_by : list of str
        Columns to group by
    
    Returns
    -------
    pd.DataFrame
        Statistics dataframe with mean, std, min, max, median, CV%
    """
    # Metrics to compute statistics for
    metrics = [
        'compression_ratio',
        'file_size_mb',
        'write_time',
        'read_time',
        'write_throughput_gbs',
        'read_throughput_gbs'
    ]
    
    # Filter to only metrics that exist
    metrics = [m for m in metrics if m in combined_df.columns]
    
    # Compute statistics
    stats_list = []
    
    for group_keys, group_df in combined_df.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        
        stats_row = dict(zip(group_by, group_keys))
        stats_row['n_runs'] = len(group_df)
        
        for metric in metrics:
            values = group_df[metric].dropna()
            
            if len(values) == 0:
                continue
            
            stats_row[f'{metric}_mean'] = values.mean()
            stats_row[f'{metric}_std'] = values.std()
            stats_row[f'{metric}_min'] = values.min()
            stats_row[f'{metric}_max'] = values.max()
            stats_row[f'{metric}_median'] = values.median()
            
            # Coefficient of variation (CV%) - only for non-zero means
            if stats_row[f'{metric}_mean'] != 0:
                cv = (stats_row[f'{metric}_std'] / stats_row[f'{metric}_mean']) * 100
                stats_row[f'{metric}_cv_percent'] = cv
            else:
                stats_row[f'{metric}_cv_percent'] = 0.0
        
        # Add chunk_size if available (should be constant across runs)
        if 'chunk_size' in group_df.columns:
            chunk_sizes = group_df['chunk_size'].dropna().unique()
            if len(chunk_sizes) == 1:
                stats_row['chunk_size'] = chunk_sizes[0]
        
        stats_list.append(stats_row)
    
    stats_df = pd.DataFrame(stats_list)
    
    return stats_df


def generate_summary_report(stats_df, combined_df, output_file):
    """Generate human-readable summary report."""
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MULTI-RUN BENCHMARK STATISTICS SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        # Overall statistics
        f.write(f"Total runs: {combined_df['run_number'].nunique()}\n")
        f.write(f"Datasets: {combined_df['dataset'].nunique()}\n")
        f.write(f"Methods per dataset: {combined_df['method'].nunique()}\n")
        f.write(f"Total measurements: {len(combined_df)}\n\n")
        
        # Variability analysis
        f.write("="*100 + "\n")
        f.write("MEASUREMENT VARIABILITY (Coefficient of Variation %)\n")
        f.write("="*100 + "\n\n")
        
        metrics_to_report = [
            ('compression_ratio', 'Compression Ratio'),
            ('write_time', 'Write Time'),
            ('read_time', 'Read Time'),
            ('write_throughput_gbs', 'Write Throughput'),
            ('read_throughput_gbs', 'Read Throughput')
        ]
        
        for metric_col, metric_name in metrics_to_report:
            cv_col = f'{metric_col}_cv_percent'
            if cv_col in stats_df.columns:
                cv_values = stats_df[cv_col].dropna()
                if len(cv_values) > 0:
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean CV: {cv_values.mean():.2f}%\n")
                    f.write(f"  Median CV: {cv_values.median():.2f}%\n")
                    f.write(f"  Max CV: {cv_values.max():.2f}%\n")
                    f.write(f"  Min CV: {cv_values.min():.2f}%\n\n")
        
        # Top methods by compression ratio
        f.write("="*100 + "\n")
        f.write("TOP 10 METHODS BY COMPRESSION RATIO (Mean ± Std)\n")
        f.write("="*100 + "\n\n")
        
        if 'compression_ratio_mean' in stats_df.columns:
            top_methods = stats_df.nlargest(10, 'compression_ratio_mean')
            
            f.write(f"{'Rank':<6} {'Dataset':<20} {'Method':<30} {'Ratio':<20} {'CV%':<10}\n")
            f.write("-"*100 + "\n")
            
            for i, (_, row) in enumerate(top_methods.iterrows(), 1):
                ratio_str = f"{row['compression_ratio_mean']:.2f} ± {row['compression_ratio_std']:.2f}"
                cv_str = f"{row['compression_ratio_cv_percent']:.2f}%"
                f.write(f"{i:<6} {row['dataset']:<20} {row['method']:<30} {ratio_str:<20} {cv_str:<10}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("NOTES\n")
        f.write("="*100 + "\n\n")
        f.write("- CV% (Coefficient of Variation) = (std / mean) × 100\n")
        f.write("- Low CV% (<5%) indicates consistent performance across runs\n")
        f.write("- High CV% (>10%) may indicate system load variability or measurement noise\n")
        f.write("- Compression ratio should have very low CV% (deterministic)\n")
        f.write("- Timing measurements typically have higher CV% (system-dependent)\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-run benchmark results and compute statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=str, default='../../results',
                       help='Base results directory containing run_XXX folders (default: ../../results)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for aggregated results (default: results_dir/aggregated)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    # Setup paths
    results_dir = Path(args.results_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'aggregated'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    if args.verbose:
        print("\n" + "="*100)
        print("AGGREGATE MULTI-RUN BENCHMARK RESULTS")
        print("="*100)
        print(f"Results directory: {results_dir.resolve()}")
        print(f"Output directory: {output_dir.resolve()}")
        print()
    
    # Find run directories
    try:
        run_dirs = find_run_directories(results_dir)
    except Exception as e:
        print(f"Error finding run directories: {e}")
        sys.exit(1)
    
    if len(run_dirs) == 0:
        print(f"No run_XXX directories found in {results_dir}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(run_dirs)} run directories:")
        for run_dir in run_dirs:
            print(f"  - {run_dir.name}")
        print()
    
    # Load all runs
    if args.verbose:
        print("Loading benchmark results from all runs...")
        print()
    
    try:
        combined_df = load_all_runs(run_dirs, verbose=args.verbose)
    except Exception as e:
        print(f"Error loading results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute statistics
    if args.verbose:
        print("\nComputing statistics...")
    
    stats_df = compute_statistics(combined_df)
    
    if args.verbose:
        print(f"Computed statistics for {len(stats_df)} dataset/method combinations")
    
    # Save results
    if args.verbose:
        print("\nSaving results...")
    
    # 1. Statistics CSV
    stats_file = output_dir / 'statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to: {stats_file}")
    
    # 2. Combined raw data
    combined_file = output_dir / 'all_runs_combined.csv'
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Combined raw data saved to: {combined_file}")
    
    # 3. Metadata
    metadata = {
        'aggregation_date': datetime.now().isoformat(),
        'n_runs': int(combined_df['run_number'].nunique()),
        'n_datasets': int(combined_df['dataset'].nunique()),
        'n_methods': int(combined_df['method'].nunique()),
        'total_measurements': len(combined_df),
        'run_directories': [str(d) for d in run_dirs],
        'datasets': sorted(combined_df['dataset'].unique().tolist()),
        'methods': sorted(combined_df['method'].unique().tolist())
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    # 4. Summary report
    summary_file = output_dir / 'summary_report.txt'
    generate_summary_report(stats_df, combined_df, summary_file)
    print(f"✓ Summary report saved to: {summary_file}")
    
    # Print summary statistics
    if args.verbose:
        print("\n" + "="*100)
        print("VARIABILITY SUMMARY")
        print("="*100)
        
        for metric in ['compression_ratio', 'write_time', 'read_time']:
            cv_col = f'{metric}_cv_percent'
            if cv_col in stats_df.columns:
                cv_values = stats_df[cv_col].dropna()
                if len(cv_values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Mean CV: {cv_values.mean():.2f}%")
                    print(f"  Median CV: {cv_values.median():.2f}%")
                    print(f"  Range: {cv_values.min():.2f}% - {cv_values.max():.2f}%")
    
    print("\n" + "="*100)
    print("AGGREGATION COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review summary_report.txt for variability analysis")
    print("  2. Use statistics.csv for figures with error bars")
    print("  3. Update visualization scripts to use aggregated data")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
