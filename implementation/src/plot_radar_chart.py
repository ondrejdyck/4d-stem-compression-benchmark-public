#!/usr/bin/env python3
"""
Radar Chart: Algorithm Performance Comparison

Creates a radar chart comparing compression algorithms across three key metrics:
- Compression ratio
- Write throughput
- Read throughput

Usage:
    python plot_radar_chart.py [--output OUTPUT_DIR]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import shared data loading functions
from data_loader import load_and_process


def create_radar_chart(df_balanced, output_dir):
    """Create radar chart comparing key algorithms.

    Parameters
    ----------
    df_balanced : pd.DataFrame
        Benchmark data filtered to balanced chunking strategy
    output_dir : Path
        Directory to save output plots
    """

    # Define the 6 key algorithms to compare
    # Selected based on top 10 performance across all three metrics
    algorithms = {
        "balanced_blosc_zlib": {"label": "Blosc Zlib", "color": "#06A77D"},
        "balanced_blosc_zstd": {"label": "Blosc Zstd", "color": "#2E86AB"},
        "balanced_blosc_lz4hc": {"label": "Blosc LZ4HC", "color": "#5E4FA2"},
        "balanced_bitshuffle_lz4": {"label": "Bitshuffle LZ4", "color": "#A23B72"},
        "balanced_blosc_lz4": {"label": "Blosc LZ4", "color": "#F18F01"},
        "balanced_blosc_blosclz": {"label": "Blosc Blosclz", "color": "#C73E1D"},
    }

    # Determine column names based on data type
    if "compression_ratio_mean" in df_balanced.columns:
        # Aggregated data
        comp_col = "compression_ratio_mean"
        write_col = "write_throughput_gbs_mean"
        read_col = "read_throughput_gbs_mean"
    else:
        # Individual run data
        comp_col = "compression_ratio"
        write_col = "write_throughput_gbs"
        read_col = "read_throughput_gbs"

    # Calculate mean metrics for each algorithm
    metrics = {}
    for algo_name, algo_info in algorithms.items():
        df_algo = df_balanced[df_balanced["method"] == algo_name]
        if len(df_algo) > 0:
            metrics[algo_name] = {
                "compression_ratio": df_algo[comp_col].mean(),
                "write_throughput": df_algo[write_col].mean(),
                "read_throughput": df_algo[read_col].mean(),
            }

    # Normalize metrics to 0-1 scale
    max_compression = max(m["compression_ratio"] for m in metrics.values())
    max_write = max(m["write_throughput"] for m in metrics.values())
    max_read = max(m["read_throughput"] for m in metrics.values())

    normalized_metrics = {}
    for algo_name, m in metrics.items():
        normalized_metrics[algo_name] = [
            m["compression_ratio"] / max_compression,
            m["write_throughput"] / max_write,
            m["read_throughput"] / max_read,
        ]

    # Create radar chart
    categories = ["Compression\nRatio", "Write\nThroughput", "Read\nThroughput"]
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

    # Plot each algorithm
    for algo_name, values in normalized_metrics.items():
        algo_info = algorithms[algo_name]
        values_plot = values + values[:1]  # Complete the circle

        ax.plot(
            angles,
            values_plot,
            "o-",
            linewidth=2.5,
            label=algo_info["label"],
            color=algo_info["color"],
        )
        ax.fill(angles, values_plot, alpha=0.15, color=algo_info["color"])

    # Customize chart
    ax.set_ylim(0, 1)

    # Place axis labels slightly further out to avoid overlap
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels=categories,
        fontsize=13,
        fontweight="bold",
    )
    # Push theta labels outward and keep them on top
    ax.tick_params(axis="x", pad=18)
    for theta, label in zip(angles[:-1], ax.get_xticklabels()):
        # Make labels extend away from the plot center
        label.set_horizontalalignment("left" if np.cos(theta) >= 0 else "right")
        label.set_verticalalignment("center")
        label.set_zorder(20)
        label.set_clip_on(False)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=15)
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # Add title and legend
    plt.title(
        "Compression Algorithm Performance Comparison\n(Normalized to Best Performer)",
        fontsize=15,
        fontweight="bold",
        pad=30,
    )

    # Larger legend text for readability at smaller display sizes.
    # Keep legend inside the axes to avoid expanding the saved figure width.
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        fontsize=14,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.8",
    )

    # Add annotation with actual values

    annotation_text = "Normalization values:\n"
    annotation_text += f"Max compression: {max_compression:.1f}×\n"
    annotation_text += f"Max write: {max_write:.2f} GiB/s\n"
    annotation_text += f"Max read: {max_read:.1f} GiB/s"

    # Keep the annotation box inside the axes to avoid expanding figure width
    ax.text(
        0.98,
        0.18,
        annotation_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=1.0, edgecolor="0.8"),
        zorder=30,
    )

    # Save figure
    plt.tight_layout()
    output_file = output_dir / "Fig2_radar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")

    # Also save as SVG for editing in Inkscape
    output_svg = output_dir / "Fig2_radar_chart.svg"
    plt.savefig(output_svg, bbox_inches="tight")
    print(f"✓ Saved: {output_svg}")

    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"{'Algorithm':<20} {'Compression':<15} {'Write (GiB/s)':<15} {'Read (GiB/s)':<15}"
    )
    print("-" * 80)
    for algo_name, m in metrics.items():
        label = algorithms[algo_name]["label"]
        print(
            f"{label:<20} {m['compression_ratio']:>6.1f}×        "
            f"{m['write_throughput']:>6.2f}          {m['read_throughput']:>6.1f}"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create radar chart comparing compression algorithms"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Results directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results)",
    )
    args = parser.parse_args()

    # Setup paths
    if args.results:
        results_dir = Path(args.results)
    else:
        # Auto-detect: script is in implementation/src/, results is in project root
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        results_dir = repo_root / "results"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = repo_root / "paper" / "generated" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RADAR CHART GENERATOR - Compression Algorithm Comparison")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data using shared data_loader module
    print("Loading benchmark data...")

    # Check if aggregated statistics are available
    aggregated_file = results_dir / "aggregated" / "statistics.csv"
    use_aggregated = aggregated_file.exists()

    df_balanced = load_and_process(
        results_dir,
        chunking_type="balanced",
        normalize=False,
        use_aggregated=use_aggregated,
    )

    if use_aggregated:
        print(
            f"✓ Loaded {len(df_balanced)} aggregated results from {df_balanced['dataset'].nunique()} datasets"
        )
        print(f"  (Based on {df_balanced['n_runs'].iloc[0]} runs per method/dataset)")
    else:
        print(
            f"✓ Loaded {len(df_balanced)} benchmark results from {df_balanced['dataset'].nunique()} datasets"
        )

    print(f"✓ Filtered to balanced chunking strategy")
    print()

    # Create radar chart
    print("Creating radar chart...")
    create_radar_chart(df_balanced, output_dir)
    print()
    print("✓ Radar chart generation complete!")


if __name__ == "__main__":
    main()
