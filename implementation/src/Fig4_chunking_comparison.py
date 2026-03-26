#!/usr/bin/env python3
"""
Visualize the effect of chunking strategy on compression performance.

This script creates a multi-panel figure showing how different HDF5 chunking
strategies affect compression ratio, write speed, and read speed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_and_process


def load_results(results_dir):
    """Load benchmark results from all datasets, using aggregated data if available."""
    results_dir = Path(results_dir)

    # Check if aggregated statistics are available
    aggregated_file = results_dir / "aggregated" / "statistics.csv"
    use_aggregated = aggregated_file.exists()

    if use_aggregated:
        print("  Using aggregated statistics (with error data)...")
        df = load_and_process(
            results_dir, chunking_type="all", normalize=False, use_aggregated=True
        )
        return df, True
    else:
        print("  Using individual run data (no error bars)...")
        df = load_and_process(results_dir, chunking_type="all", normalize=False)
        return df, False


def extract_chunking_data(df):
    """
    Extract base algorithm and chunking strategy from method names.

    Method names follow pattern: {chunk_strategy}_{algorithm}
    e.g., "real_space_gzip_6", "balanced_blosc_zstd", "single_frame_lz4_hdf5"
    """
    # Filter to only HDF5 methods with chunking info
    df = df[
        df["method"].str.contains("real_space|balanced|single_frame", na=False)
    ].copy()

    # Extract chunking strategy and base algorithm
    def parse_method(method):
        parts = method.split("_")
        if parts[0] == "real":
            chunk_strategy = "real_space"
            base_algo = "_".join(parts[2:])
        elif parts[0] == "balanced":
            chunk_strategy = "balanced"
            base_algo = "_".join(parts[1:])
        elif parts[0] == "single":
            chunk_strategy = "single_frame"
            base_algo = "_".join(parts[2:])
        else:
            chunk_strategy = None
            base_algo = method

        return pd.Series(
            {"chunk_strategy": chunk_strategy, "base_algorithm": base_algo}
        )

    df[["chunk_strategy", "base_algorithm"]] = df["method"].apply(parse_method)

    # Remove rows where parsing failed
    df = df[df["chunk_strategy"].notna()].copy()

    return df


def select_representative_algorithms(df, n_algorithms=6):
    """
    Select the 6 key algorithms that perform well across all metrics.

    These algorithms were selected based on appearing in top 10 for compression,
    write speed, and read speed (plus blosc_zlib for highest compression).
    """
    # The 6 key algorithms identified from multi-run analysis
    selected = [
        "blosc_zlib",  # Highest compression ratio (13.49×)
        "blosc_zstd",  # Best overall balance (13.47×)
        "blosc_lz4hc",  # High compression + fast read (10.08×)
        "bitshuffle_lz4",  # Well-balanced (8.95×)
        "blosc_lz4",  # Fast compression/decompression (7.88×)
        "blosc_blosclz",  # Balanced performance (7.06×)
    ]

    # Verify all are available
    available = df["base_algorithm"].unique()
    selected = [algo for algo in selected if algo in available]

    return selected


def create_chunking_comparison_plot(
    df, output_path, dataset_name=None, use_aggregated=False
):
    """
    Create multi-panel figure showing chunking effects.

    Three panels:
    - A) Compression ratio
    - B) Write throughput (GiB/s)
    - C) Read throughput (GiB/s)
    """
    # Select representative algorithms
    selected_algos = select_representative_algorithms(df, n_algorithms=6)
    df_plot = df[df["base_algorithm"].isin(selected_algos)].copy()

    # Define chunking strategy order and labels
    chunk_order = ["real_space", "balanced", "single_frame"]
    chunk_labels = {
        "real_space": "Real-space\n(32×32×...)",
        "balanced": "Balanced\n(16×16×...)",
        "single_frame": "Single-frame\n(1×1×...)",
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Color scheme for chunking strategies
    colors = {
        "real_space": "#2E86AB",  # Blue
        "balanced": "#A23B72",  # Purple
        "single_frame": "#F18F01",  # Orange
    }

    # Panel A: Compression Ratio (no error bars - compression is deterministic)
    ax = axes[0]
    plot_grouped_bars(
        ax,
        df_plot,
        "compression_ratio",
        selected_algos,
        chunk_order,
        colors,
        chunk_labels,
        use_aggregated,
        show_error_bars=False,
    )
    ax.set_ylabel("Compression Ratio", fontsize=12, fontweight="bold")
    ax.set_title("A) Compression Performance", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel B: Write Throughput (with error bars if aggregated)
    ax = axes[1]
    plot_grouped_bars(
        ax,
        df_plot,
        "write_throughput_gbs",
        selected_algos,
        chunk_order,
        colors,
        chunk_labels,
        use_aggregated,
        show_error_bars=use_aggregated,
    )
    ax.set_ylabel("Write Throughput (GiB/s)", fontsize=12, fontweight="bold")
    ax.set_title("B) Write Throughput", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel C: Read Throughput (with error bars if aggregated)
    ax = axes[2]
    plot_grouped_bars(
        ax,
        df_plot,
        "read_throughput_gbs",
        selected_algos,
        chunk_order,
        colors,
        chunk_labels,
        use_aggregated,
        show_error_bars=use_aggregated,
    )
    ax.set_ylabel("Read Throughput (GiB/s)", fontsize=12, fontweight="bold")
    ax.set_title("C) Read Throughput", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add legend to upper left of middle panel (panel B)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=colors[cs], edgecolor="black", linewidth=0.5
        )
        for cs in chunk_order
    ]
    labels = [chunk_labels[cs] for cs in chunk_order]
    axes[1].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        fontsize=11,
        framealpha=0.95,
        title="Chunking Strategy",
        title_fontsize=12,
    )

    # Overall title
    title = "Effect of Chunking Strategy on Compression Performance"
    if dataset_name:
        title += f"\nDataset: {dataset_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    return fig


def plot_grouped_bars(
    ax,
    df,
    metric,
    algorithms,
    chunk_order,
    colors,
    chunk_labels,
    use_aggregated=False,
    show_error_bars=True,
):
    """Plot grouped bar chart for a single metric."""
    x = np.arange(len(algorithms))
    width = 0.25

    # Determine metric column name based on data type
    if use_aggregated:
        metric_col = f"{metric}_mean"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"
    else:
        metric_col = metric

    for i, chunk_strategy in enumerate(chunk_order):
        values = []
        errors_lower = []
        errors_upper = []

        for algo in algorithms:
            subset = df[
                (df["base_algorithm"] == algo)
                & (df["chunk_strategy"] == chunk_strategy)
            ]
            if len(subset) > 0:
                # Average across datasets if multiple
                mean_val = subset[metric_col].mean()
                values.append(mean_val)

                # Calculate error bars if using aggregated data
                if use_aggregated and show_error_bars:
                    min_val = subset[min_col].mean()
                    max_val = subset[max_col].mean()
                    errors_lower.append(max(0, mean_val - min_val))
                    errors_upper.append(max(0, max_val - mean_val))
                else:
                    errors_lower.append(0)
                    errors_upper.append(0)
            else:
                values.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        offset = (i - 1) * width

        # Plot bars with error bars if applicable
        if use_aggregated and show_error_bars:
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=chunk_labels[chunk_strategy],
                color=colors[chunk_strategy],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                yerr=[errors_lower, errors_upper],
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 1.5,
                    "capsize": 3,
                    "capthick": 1.5,
                },
            )
        else:
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=chunk_labels[chunk_strategy],
                color=colors[chunk_strategy],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )

        # Value labels removed for cleaner appearance

    # Format x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        [algo.replace("_", "\n") for algo in algorithms], fontsize=10, rotation=0
    )
    ax.set_xlim(-0.5, len(algorithms) - 0.5)

    # Format y-axis
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add space for labels


def print_chunking_summary(df):
    """Print summary statistics about chunking effects."""
    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 70)

    # Determine column names based on data type
    if "compression_ratio_mean" in df.columns:
        # Aggregated data
        comp_col = "compression_ratio_mean"
        write_col = "write_throughput_gbs_mean"
        read_col = "read_throughput_gbs_mean"
    else:
        # Individual run data
        comp_col = "compression_ratio"
        write_col = "write_throughput_gbs"
        read_col = "read_throughput_gbs"

    # Group by base algorithm and chunking strategy
    grouped = (
        df.groupby(["base_algorithm", "chunk_strategy"])
        .agg({comp_col: "mean", write_col: "mean", read_col: "mean"})
        .round(2)
    )

    # Rename columns for display
    grouped.columns = [
        "compression_ratio",
        "write_throughput_gbs",
        "read_throughput_gbs",
    ]

    print("\nAverage performance by algorithm and chunking strategy:")
    print(grouped.to_string())

    # Calculate percentage differences
    print("\n" + "-" * 70)
    print("Impact of chunking strategy (% change from balanced):")
    print("-" * 70)

    for algo in df["base_algorithm"].unique():
        if "none" in algo:
            continue

        algo_data = df[df["base_algorithm"] == algo]

        balanced = algo_data[algo_data["chunk_strategy"] == "balanced"]
        real_space = algo_data[algo_data["chunk_strategy"] == "real_space"]
        single_frame = algo_data[algo_data["chunk_strategy"] == "single_frame"]

        if len(balanced) == 0:
            continue

        print(f"\n{algo}:")

        if len(real_space) > 0:
            comp_diff = (
                real_space[comp_col].mean() / balanced[comp_col].mean() - 1
            ) * 100
            read_diff = (
                real_space[read_col].mean() / balanced[read_col].mean() - 1
            ) * 100
            print(
                f"  Real-space vs Balanced: {comp_diff:+.1f}% compression, {read_diff:+.1f}% read speed"
            )

        if len(single_frame) > 0:
            comp_diff = (
                single_frame[comp_col].mean() / balanced[comp_col].mean() - 1
            ) * 100
            read_diff = (
                single_frame[read_col].mean() / balanced[read_col].mean() - 1
            ) * 100
            print(
                f"  Single-frame vs Balanced: {comp_diff:+.1f}% compression, {read_diff:+.1f}% read speed"
            )


def main():
    """Main execution function."""
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    results_dir = repo_root / "results"
    output_dir = repo_root / "paper" / "generated" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Fig4_chunking_comparison"

    print("Loading benchmark results...")
    df, use_aggregated = load_results(results_dir)
    print(f"Loaded {len(df)} total results from {df['dataset'].nunique()} datasets")
    if use_aggregated:
        print(f"  (Based on {df['n_runs'].iloc[0]} runs per method/dataset)")

    print("\nExtracting chunking information...")
    df_chunking = extract_chunking_data(df)
    print(f"Found {len(df_chunking)} results with chunking strategies")
    print(f"Base algorithms: {sorted(df_chunking['base_algorithm'].unique())}")

    # Print summary statistics
    print_chunking_summary(df_chunking)

    # Create visualization
    print("\nCreating chunking comparison plot...")
    if use_aggregated:
        print("  Including error bars on panels B & C (min-max range)")
    create_chunking_comparison_plot(
        df_chunking, output_file, use_aggregated=use_aggregated
    )

    print("\n" + "=" * 70)
    print("CHUNKING ANALYSIS COMPLETE")
    print("=" * 70)
    if use_aggregated:
        print("Figure includes error bars on write/read speed panels")


if __name__ == "__main__":
    main()
