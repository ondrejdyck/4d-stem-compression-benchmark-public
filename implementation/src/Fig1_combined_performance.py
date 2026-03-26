#!/usr/bin/env python3
"""
Combined Performance Figure for Paper

Creates a 3-panel figure stacked vertically showing:
- Panel A: Compression ratios across datasets
- Panel B: Write throughput across datasets
- Panel C: Read throughput across datasets

Uses the same style as individual cross-dataset plots (viridis colors, gray mean bars).

Usage:
    python Fig1_combined_performance.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_and_process, get_error_bars


def create_panel(
    ax, df, metric, ylabel, title, top_n=10, use_aggregated=False, show_error_bars=True
):
    """
    Create a single panel with the same style as original plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df : pd.DataFrame
        Benchmark data (either individual runs or aggregated statistics)
    metric : str
        Column name to plot ('compression_ratio', 'write_throughput_gbs', 'read_throughput_gbs')
    ylabel : str
        Y-axis label
    title : str
        Panel title
    top_n : int
        Number of top methods to show
    use_aggregated : bool
        If True, df contains aggregated statistics with _mean, _std, _min, _max columns
    show_error_bars : bool
        If True and use_aggregated=True, show error bars (default: True)
    """

    # Determine actual column names based on data type
    if use_aggregated:
        metric_col = f"{metric}_mean"
        # For sparsity, we need to get it from metadata (not in aggregated stats)
        # We'll need to load it separately or pass it in
        # For now, we'll extract sparsity from the first occurrence
    else:
        metric_col = metric

    # Get unique datasets and their sparsity
    if use_aggregated:
        if "sparsity" in df.columns:
            dataset_info = (
                df.groupby("dataset")
                .agg({metric_col: "mean", "sparsity": "first"})
                .reset_index()
            )
        else:
            dataset_info = df.groupby("dataset").agg({metric_col: "mean"}).reset_index()
            dataset_info["sparsity"] = 0.0
    else:
        dataset_info = (
            df.groupby("dataset")
            .agg({"sparsity": "first", metric: "mean"})
            .reset_index()
        )

    # Sort datasets by sparsity (descending) for consistent ordering
    dataset_order = dataset_info.sort_values("sparsity", ascending=False)[
        "dataset"
    ].values

    # Calculate mean metric for each method across all datasets
    method_means = df.groupby("method")[metric_col].mean().sort_values(ascending=False)

    # Select top N methods
    top_methods = method_means.head(top_n).index.tolist()

    # Filter to top methods
    df_filtered = df[df["method"].isin(top_methods)].copy()

    # Create pivot table for values
    pivot = df_filtered.pivot(index="method", columns="dataset", values=metric_col)

    # Create pivot tables for error bars if using aggregated data
    if use_aggregated:
        pivot_min = df_filtered.pivot(
            index="method", columns="dataset", values=f"{metric}_min"
        )
        pivot_max = df_filtered.pivot(
            index="method", columns="dataset", values=f"{metric}_max"
        )

    # Reorder columns by sparsity (most sparse first)
    pivot = pivot[dataset_order]
    if use_aggregated:
        pivot_min = pivot_min[dataset_order]
        pivot_max = pivot_max[dataset_order]

    # Reorder rows by mean metric (best at top)
    # Note: For horizontal bars, top of chart = bottom of index, so we reverse
    row_order = method_means[method_means.index.isin(top_methods)].index[::-1]
    pivot = pivot.loc[row_order]
    if use_aggregated:
        pivot_min = pivot_min.loc[row_order]
        pivot_max = pivot_max.loc[row_order]

    # Save original dataset and method names for error bar lookups
    original_datasets = pivot.columns.copy()
    original_methods = pivot.index.copy()

    # Create prettier dataset labels
    dataset_labels = {}
    for dataset in dataset_order:
        sparsity = dataset_info[dataset_info["dataset"] == dataset]["sparsity"].values[
            0
        ]

        if "4D_EELS" in dataset:
            label = f"4D EELS ({sparsity * 100:.1f}% sparse)"
        elif "3D_EELS" in dataset:
            label = f"3D EELS ({sparsity * 100:.1f}% sparse)"
        elif "4D_Diff" in dataset:
            if "2x2-binning" in dataset:
                label = f"4D Diff. 2×2 bin ({sparsity * 100:.1f}% sparse)"
            elif "4x4-binning" in dataset:
                label = f"4D Diff. 4×4 bin ({sparsity * 100:.1f}% sparse)"
            else:
                label = f"4D Diff. ({sparsity * 100:.1f}% sparse)"
        else:
            label = f"{dataset} ({sparsity * 100:.1f}% sparse)"

        dataset_labels[dataset] = label

    # Rename columns with prettier labels
    pivot.columns = [dataset_labels[col] for col in pivot.columns]

    # Create prettier method labels
    method_labels = {}
    for method in pivot.index:
        # Remove chunking prefix
        clean = (
            method.replace("balanced_", "")
            .replace("real_space_", "")
            .replace("single_frame_", "")
        )

        # Capitalize and format
        if "blosc_zstd" in clean:
            label = "Blosc Zstd"
        elif "blosc_zlib" in clean:
            label = "Blosc Zlib"
        elif "blosc_lz4hc" in clean:
            label = "Blosc LZ4HC"
        elif "blosc_lz4" in clean:
            label = "Blosc LZ4"
        elif "gzip_9" in clean:
            label = "Gzip-9"
        elif "gzip_1" in clean:
            label = "Gzip-1"
        elif "lzf" in clean:
            label = "LZF"
        elif "szip" in clean:
            label = "Szip"
        else:
            label = clean.replace("_", " ").title()

        method_labels[method] = label

    pivot.index = [method_labels[idx] for idx in pivot.index]

    # Calculate mean for each algorithm (in pivot order)
    pivot_means = pivot.mean(axis=1)

    # Add background shading for mean values FIRST (so it's behind everything)
    bar_height = 0.75  # Same as bar width

    for i, (idx, mean_val) in enumerate(pivot_means.items()):
        y_pos = i

        # Add subtle rectangular patch from 0 to mean value
        ax.barh(
            y_pos,
            mean_val,
            height=bar_height,
            color="lightgray",
            alpha=0.3,
            zorder=1,
            edgecolor="gray",
            linewidth=1,
            linestyle="--",
        )

    # Define color palette (ordered by sparsity) - same viridis as original
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pivot.columns)))

    # Plot grouped horizontal bars ON TOP of background shading
    pivot.plot(
        kind="barh",
        ax=ax,
        width=0.75,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
        legend=False,
    )

    # Add error bars if using aggregated data and requested
    if use_aggregated and show_error_bars:
        # Calculate error bar positions and values
        n_methods = len(pivot)
        n_datasets = len(pivot.columns)
        bar_height = 0.75

        # For grouped horizontal bars, bars are positioned at:
        # y = method_index + (dataset_index - n_datasets/2 + 0.5) * (bar_height / n_datasets)
        for method_idx, (pretty_method, orig_method) in enumerate(
            zip(pivot.index, original_methods)
        ):
            for dataset_idx, (pretty_dataset, orig_dataset) in enumerate(
                zip(pivot.columns, original_datasets)
            ):
                value = pivot.loc[pretty_method, pretty_dataset]
                min_val = pivot_min.loc[orig_method, orig_dataset]
                max_val = pivot_max.loc[orig_method, orig_dataset]

                # Calculate y position (same as pandas bar plot)
                y_pos = method_idx + (dataset_idx - n_datasets / 2 + 0.5) * (
                    bar_height / n_datasets
                )

                # Calculate error bar lengths (ensure non-negative)
                lower_err = max(0, value - min_val)
                upper_err = max(0, max_val - value)

                # Only draw error bar if there's actual variation
                if lower_err > 0 or upper_err > 0:
                    ax.errorbar(
                        value,
                        y_pos,
                        xerr=[[lower_err], [upper_err]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.5,
                        capsize=3,
                        capthick=1.5,
                        zorder=4,
                    )

    # Formatting - LARGE fonts for publication
    ax.set_xlabel(ylabel, fontsize=22, fontweight="bold")
    ax.set_ylabel("Compression Algorithm", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=15)

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=20)

    # Grid
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    ax.set_axisbelow(True)

    return pivot.columns  # Return dataset labels for legend


def create_combined_figure(df, output_path, top_n=10, use_aggregated=False):
    """
    Create 3-panel combined figure stacked vertically.

    Parameters
    ----------
    df : pd.DataFrame
        Combined benchmark data (individual runs or aggregated statistics)
    output_path : Path
        Output file path (without extension)
    top_n : int
        Number of top methods to show
    use_aggregated : bool
        If True, df contains aggregated statistics with error bars
    """

    # Create figure with 3 panels stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(16, 24))

    # Panel A: Compression Ratio (no error bars - compression is deterministic)
    dataset_labels = create_panel(
        axes[0],
        df,
        "compression_ratio",
        "Compression Ratio",
        "A) Compression Performance Across Datasets (top 10)",
        top_n,
        use_aggregated,
        show_error_bars=False,
    )

    # Panel B: Write Throughput (with error bars if aggregated)
    create_panel(
        axes[1],
        df,
        "write_throughput_gbs",
        "Write Throughput (GiB/s)",
        "B) Write Speed Performance Across Datasets (top 10)",
        top_n,
        use_aggregated,
        show_error_bars=use_aggregated,
    )

    # Panel C: Read Throughput (with error bars if aggregated)
    create_panel(
        axes[2],
        df,
        "read_throughput_gbs",
        "Read Throughput (GiB/s)",
        "C) Read Speed Performance Across Datasets (top 10)",
        top_n,
        use_aggregated,
        show_error_bars=use_aggregated,
    )

    # Add legend inside the third panel (lower right corner)
    # Get colors from viridis
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dataset_labels)))

    # Create legend handles
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5) for color in colors
    ]

    # Add mean bar to legend
    mean_patch = Patch(
        facecolor="lightgray",
        edgecolor="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.3,
        label="Mean across datasets",
    )
    handles.append(mean_patch)
    labels = list(dataset_labels) + ["Mean across datasets"]

    # Place legend in lower right of third panel
    axes[2].legend(
        handles,
        labels,
        title="Dataset (ordered by sparsity)",
        loc="lower right",
        fontsize=18,
        title_fontsize=20,
        framealpha=0.95,
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close()


def main():
    """Main execution."""
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    results_dir = repo_root / "results"
    output_dir = repo_root / "paper" / "generated" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Fig1_combined_performance"

    # Load dataset inventory for sparsity labels (public, CSV-based source of truth)
    inventory_file = repo_root / "results" / "dataset_inventory.csv"
    inventory = pd.read_csv(inventory_file)[["dataset_id", "sparsity_fraction"]]
    inventory = inventory.rename(
        columns={"dataset_id": "dataset", "sparsity_fraction": "sparsity"}
    )

    # Check if aggregated statistics are available
    aggregated_file = results_dir / "aggregated" / "statistics.csv"
    use_aggregated = aggregated_file.exists()

    if use_aggregated:
        print("Loading aggregated benchmark statistics (with error bars)...")
        df = load_and_process(
            results_dir, chunking_type="balanced", normalize=False, use_aggregated=True
        )
        df = df.merge(inventory, on="dataset", how="left", suffixes=("", "_inv"))
        if "sparsity_inv" in df.columns:
            df["sparsity"] = df["sparsity"].fillna(df["sparsity_inv"])
            df = df.drop(columns=["sparsity_inv"])
        print(
            f"Loaded {len(df)} aggregated results from {df['dataset'].nunique()} datasets"
        )
        print(f"  (Based on {df['n_runs'].iloc[0]} runs per method/dataset)")
    else:
        print("Loading individual benchmark results (no error bars)...")
        df = load_and_process(results_dir, chunking_type="balanced", normalize=False)
        print(f"Loaded {len(df)} results from {df['dataset'].nunique()} datasets")
        print("  Note: Run aggregate_multi_run_results.py to enable error bars")

    print("\nCreating combined 3-panel figure (stacked vertically)...")
    create_combined_figure(df, output_file, top_n=10, use_aggregated=use_aggregated)

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_file}.pdf")
    print("\nFigure features:")
    print("  - Viridis color palette")
    print("  - Gray bars showing mean values across datasets")
    if use_aggregated:
        print("  - Black error bars on panels B & C (min-max range across 10 runs)")
        print("  - No error bars on panel A (compression is deterministic)")
    print("  - Large, readable fonts (18-24pt)")
    print("  - 3 panels stacked vertically")
    print("  - Publication-quality vector PDF")


if __name__ == "__main__":
    main()
