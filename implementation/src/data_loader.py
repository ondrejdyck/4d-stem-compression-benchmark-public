#!/usr/bin/env python3
"""
Data Loader Module for 4D STEM Compression Benchmark

Provides shared functions for loading and processing benchmark results.
All visualization scripts should use these functions to ensure consistency.

Functions:
    load_all_benchmarks() - Load all benchmark CSVs with metadata
    calculate_metrics() - Add derived metrics (throughput, etc.)
    get_algorithm_family() - Categorize algorithms by family
    normalize_metrics() - Normalize metrics to 0-1 scale
    filter_by_chunking() - Filter to specific chunking strategy
    get_best_per_dataset() - Get best performing method per dataset
"""

import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path


def load_all_benchmarks(results_dir):
    """
    Load all benchmark results and merge with metadata.

    Parameters
    ----------
    results_dir : Path or str
        Directory containing benchmark results (with subdirectories per dataset)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        - All original CSV columns (method, compression_ratio, file_size_mb, etc.)
        - dataset: dataset name
        - sparsity: fraction of zeros
        - data_size_gb: original data size in GiB
        - data_shape: shape of 4D array
        - data_dtype: data type
        - unique_values: number of unique values
        - max_value: maximum value in data
        - mean_value: mean value in data
    """
    results_dir = Path(results_dir)
    all_data = []

    # Find all metadata files
    metadata_files = glob.glob(str(results_dir / "*/metadata.json"))

    if len(metadata_files) == 0:
        raise FileNotFoundError(f"No metadata.json files found in {results_dir}")

    for metadata_file in metadata_files:
        # Load metadata
        with open(metadata_file) as f:
            meta = json.load(f)

        # Load corresponding CSV
        csv_file = metadata_file.replace("metadata.json", "benchmark_results.csv")
        if not Path(csv_file).exists():
            print(f"Warning: CSV not found for {meta['dataset_name']}, skipping")
            continue

        df = pd.read_csv(csv_file)

        # Add metadata columns
        df["dataset"] = meta["dataset_name"]
        df["sparsity"] = meta["sparsity"]
        df["data_size_gb"] = meta["original_size_mb"] / 1024
        df["data_shape"] = str(meta["data_shape"])
        df["data_dtype"] = meta["data_dtype"]
        df["unique_values"] = meta["unique_values"]
        df["max_value"] = meta["max_value"]
        df["mean_value"] = meta["mean_value"]

        all_data.append(df)

    # Combine all datasets
    df_combined = pd.concat(all_data, ignore_index=True)

    return df_combined


def calculate_metrics(df):
    """
    Add derived metrics to benchmark DataFrame.

    Adds the following columns:
    - write_throughput_gbs: Write throughput in GiB/s
    - read_throughput_gbs: Read throughput in GiB/s
    - algorithm_family: Algorithm family (Blosc, Gzip, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_all_benchmarks()

    Returns
    -------
    pd.DataFrame
        DataFrame with additional metric columns
    """
    df = df.copy()

    # Calculate throughput (GiB/s)
    # Handle division by zero and NaN
    df["write_throughput_gbs"] = df.apply(
        lambda row: row["data_size_gb"] / row["write_time"]
        if pd.notna(row["write_time"]) and row["write_time"] > 0
        else np.nan,
        axis=1,
    )

    df["read_throughput_gbs"] = df.apply(
        lambda row: row["data_size_gb"] / row["read_time"]
        if pd.notna(row["read_time"]) and row["read_time"] > 0
        else np.nan,
        axis=1,
    )

    # Add algorithm family
    df["algorithm_family"] = df["method"].apply(get_algorithm_family)

    return df


def get_algorithm_family(method_name):
    """
    Categorize compression algorithm by family.

    Parameters
    ----------
    method_name : str
        Method name (e.g., 'balanced_blosc_zstd')

    Returns
    -------
    str
        Algorithm family: 'Blosc', 'Gzip', 'LZF', 'Szip', 'LZ4', 'Bitshuffle', 'Sparse', 'Custom', 'None'
    """
    method_lower = str(method_name).lower()

    if "blosc" in method_lower:
        return "Blosc"
    elif "gzip" in method_lower:
        return "Gzip"
    elif "lzf" in method_lower:
        return "LZF"
    elif "szip" in method_lower:
        return "Szip"
    elif "lz4" in method_lower and "blosc" not in method_lower:
        return "LZ4"
    elif "bitshuffle" in method_lower:
        return "Bitshuffle"
    elif "sparse" in method_lower:
        return "Sparse"
    elif "uint8" in method_lower or "simple" in method_lower:
        return "Custom"
    elif "none" in method_lower:
        return "None"
    else:
        return "Other"


def normalize_metrics(df, metrics=None, method="minmax"):
    """
    Normalize specified metrics to 0-1 scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics to normalize
    metrics : list of str, optional
        Metrics to normalize. Default: ['compression_ratio', 'write_throughput_gbs', 'read_throughput_gbs']
    method : str, optional
        Normalization method: 'minmax' (default) or 'zscore'

    Returns
    -------
    pd.DataFrame
        DataFrame with additional normalized_* columns
    """
    if metrics is None:
        metrics = ["compression_ratio", "write_throughput_gbs", "read_throughput_gbs"]

    df = df.copy()

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame, skipping")
            continue

        if method == "minmax":
            # Normalize to 0-1 using min-max scaling
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df[f"normalized_{metric}"] = (df[metric] - min_val) / (
                    max_val - min_val
                )
            else:
                df[f"normalized_{metric}"] = 0.0

        elif method == "zscore":
            # Normalize using z-score
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            if std_val > 0:
                df[f"normalized_{metric}"] = (df[metric] - mean_val) / std_val
            else:
                df[f"normalized_{metric}"] = 0.0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return df


def filter_by_chunking(df, chunking_type="balanced"):
    """
    Filter DataFrame to specific chunking strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with benchmark results
    chunking_type : str
        Chunking type: 'balanced', 'real_space', 'single_frame', or 'all'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if chunking_type == "all":
        return df.copy()

    # Filter by method name prefix
    if chunking_type == "balanced":
        return df[df["method"].str.contains("balanced", na=False)].copy()
    elif chunking_type == "real_space":
        return df[df["method"].str.contains("real_space", na=False)].copy()
    elif chunking_type == "single_frame":
        return df[df["method"].str.contains("single_frame", na=False)].copy()
    else:
        raise ValueError(f"Unknown chunking type: {chunking_type}")


def get_best_per_dataset(df, metric="compression_ratio", chunking_type="balanced"):
    """
    Get best performing method for each dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with benchmark results
    metric : str
        Metric to optimize: 'compression_ratio', 'write_throughput_gbs', 'read_throughput_gbs'
    chunking_type : str
        Chunking strategy to consider: 'balanced', 'real_space', 'single_frame', 'all'

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per dataset (best method)
    """
    # Filter by chunking if specified
    if chunking_type != "all":
        df = filter_by_chunking(df, chunking_type)

    # Get best method per dataset
    df_best = df.loc[df.groupby("dataset")[metric].idxmax()]

    return df_best.reset_index(drop=True)


def get_summary_statistics(df, group_by="algorithm_family"):
    """
    Calculate summary statistics grouped by algorithm family or other column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with benchmark results
    group_by : str
        Column to group by (default: 'algorithm_family')

    Returns
    -------
    pd.DataFrame
        Summary statistics with mean, std, min, max for each metric
    """
    metrics = ["compression_ratio", "write_throughput_gbs", "read_throughput_gbs"]

    # Filter to metrics that exist
    available_metrics = [m for m in metrics if m in df.columns]

    summary = df.groupby(group_by)[available_metrics].agg(["mean", "std", "min", "max"])

    return summary


def load_aggregated_statistics(results_dir):
    """
    Load aggregated statistics from multi-run benchmarks.

    This function loads the statistics.csv file generated by aggregate_multi_run_results.py,
    which contains mean, std, min, max, median, and CV% for all metrics across multiple runs.

    Parameters
    ----------
    results_dir : Path or str
        Base results directory containing aggregated/ subdirectory

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - dataset, method, n_runs
        - For each metric: *_mean, *_std, *_min, *_max, *_median, *_cv_percent
        - chunk_size (if available)

    Raises
    ------
    FileNotFoundError
        If aggregated/statistics.csv does not exist
    """
    results_dir = Path(results_dir)
    stats_file = results_dir / "aggregated" / "statistics.csv"

    if not stats_file.exists():
        raise FileNotFoundError(
            f"Aggregated statistics not found: {stats_file}\n"
            f"Run aggregate_multi_run_results.py first to generate statistics from multiple runs."
        )

    df = pd.read_csv(stats_file)

    return df


def get_error_bars(df_stats, metric, error_type="minmax"):
    """
    Extract error bar data from aggregated statistics.

    Parameters
    ----------
    df_stats : pd.DataFrame
        DataFrame from load_aggregated_statistics()
    metric : str
        Base metric name (e.g., 'compression_ratio', 'write_time', 'read_throughput_gbs')
    error_type : str
        Type of error bars:
        - 'minmax': Use min and max values (shows full range)
        - 'std': Use mean ± std (standard error bars)
        - 'sem': Use mean ± std/sqrt(n) (standard error of mean)

    Returns
    -------
    tuple of (values, lower_errors, upper_errors)
        - values: Mean values for the metric
        - lower_errors: Distance from mean to lower error bar
        - upper_errors: Distance from mean to upper error bar

    Notes
    -----
    For matplotlib errorbar(), use:
        plt.errorbar(x, values, yerr=[lower_errors, upper_errors])

    For matplotlib bar(), use:
        plt.bar(x, values, yerr=[lower_errors, upper_errors])
    """
    mean_col = f"{metric}_mean"

    if mean_col not in df_stats.columns:
        raise ValueError(f"Metric '{metric}' not found in statistics DataFrame")

    values = df_stats[mean_col].values

    if error_type == "minmax":
        # Use full range (min to max)
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"

        if min_col not in df_stats.columns or max_col not in df_stats.columns:
            raise ValueError(f"Min/max columns not found for metric '{metric}'")

        lower_errors = values - df_stats[min_col].values
        upper_errors = df_stats[max_col].values - values

    elif error_type == "std":
        # Use mean ± std
        std_col = f"{metric}_std"

        if std_col not in df_stats.columns:
            raise ValueError(f"Std column not found for metric '{metric}'")

        std_values = df_stats[std_col].values
        lower_errors = std_values
        upper_errors = std_values

    elif error_type == "sem":
        # Use mean ± std/sqrt(n) (standard error of mean)
        std_col = f"{metric}_std"

        if std_col not in df_stats.columns or "n_runs" not in df_stats.columns:
            raise ValueError(f"Std or n_runs column not found for metric '{metric}'")

        sem_values = df_stats[std_col].values / np.sqrt(df_stats["n_runs"].values)
        lower_errors = sem_values
        upper_errors = sem_values

    else:
        raise ValueError(
            f"Unknown error_type: {error_type}. Use 'minmax', 'std', or 'sem'"
        )

    return values, lower_errors, upper_errors


# Convenience function for common workflow
def load_and_process(
    results_dir, chunking_type="balanced", normalize=False, use_aggregated=False
):
    """
    Load benchmark data and apply common processing steps.

    This is a convenience function that combines:
    1. load_all_benchmarks() or load_aggregated_statistics()
    2. calculate_metrics() (if not using aggregated)
    3. filter_by_chunking() (optional)
    4. normalize_metrics() (optional)

    Parameters
    ----------
    results_dir : Path or str
        Directory containing benchmark results
    chunking_type : str
        Chunking strategy: 'balanced', 'real_space', 'single_frame', 'all'
    normalize : bool
        Whether to normalize metrics to 0-1 scale
    use_aggregated : bool
        If True, load aggregated statistics (mean, std, min, max from multiple runs)
        If False, load individual run CSVs (default behavior)

    Returns
    -------
    pd.DataFrame
        Processed DataFrame ready for visualization
    """
    if use_aggregated:
        # Load aggregated statistics (already has mean values and error data)
        df = load_aggregated_statistics(results_dir)

        # Aggregated data already has throughput columns with _mean suffix
        # No need to calculate metrics

    else:
        # Load individual run data
        df = load_all_benchmarks(results_dir)

        # Calculate metrics
        df = calculate_metrics(df)

    # Filter by chunking
    if chunking_type != "all":
        if use_aggregated:
            # For aggregated data, filter by method name
            if chunking_type == "balanced":
                df = df[df["method"].str.contains("balanced", na=False)].copy()
            elif chunking_type == "real_space":
                df = df[df["method"].str.contains("real_space", na=False)].copy()
            elif chunking_type == "single_frame":
                df = df[df["method"].str.contains("single_frame", na=False)].copy()
        else:
            df = filter_by_chunking(df, chunking_type)

    # Normalize if requested
    if normalize:
        if use_aggregated:
            # For aggregated data, normalize the _mean columns
            metrics_to_normalize = []
            for base_metric in [
                "compression_ratio",
                "write_throughput_gbs",
                "read_throughput_gbs",
            ]:
                mean_col = f"{base_metric}_mean"
                if mean_col in df.columns:
                    # Temporarily rename for normalization
                    df[base_metric] = df[mean_col]
                    metrics_to_normalize.append(base_metric)

            df = normalize_metrics(df, metrics=metrics_to_normalize)

            # Clean up temporary columns
            for metric in metrics_to_normalize:
                df.drop(columns=[metric], inplace=True)
        else:
            df = normalize_metrics(df)

    return df


if __name__ == "__main__":
    # Test the module
    import sys

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent.parent.parent / "results"

    print("=" * 80)
    print("DATA LOADER MODULE TEST")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()

    # Load data
    print("Loading benchmarks...")
    df = load_all_benchmarks(results_dir)
    print(f"✓ Loaded {len(df)} results from {df['dataset'].nunique()} datasets")
    print()

    # Calculate metrics
    print("Calculating metrics...")
    df = calculate_metrics(df)
    print(
        f"✓ Added columns: {[c for c in df.columns if 'throughput' in c or 'family' in c]}"
    )
    print()

    # Show summary
    print("Algorithm family summary:")
    print(get_summary_statistics(df))
    print()

    # Show best per dataset
    print("Best method per dataset (balanced chunking):")
    df_best = get_best_per_dataset(df, chunking_type="balanced")
    for _, row in df_best.iterrows():
        print(
            f"  {row['dataset']:40s}: {row['compression_ratio']:6.1f}× ({row['method']})"
        )

    print()
    print("✓ Data loader module test complete!")
