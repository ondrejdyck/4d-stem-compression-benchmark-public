#!/usr/bin/env python3
"""
Figure 3: sparsity against compression ratio.

Shows the three datasets stored as uint16 only. The binned datasets are stored
as float32; compression ratio is not comparable across container widths, and the
16-bit binary-entropy bound does not govern them.

Usage:
    uv run python implementation/src/plot_sparsity_compression.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Embed TrueType rather than matplotlib's default Type 3 fonts. Type 3 is
# rejected by several journals' production systems and carries no ToUnicode
# map, so text in the figure cannot be selected, searched or read aloud.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# Dataset sparsity values (from DATASET_INVENTORY.md)
DATASET_SPARSITY = {
    "3D_EELS": 0.495,
    "4D_EELS": 0.928,
    "4D_Diff": 0.747,
    "4D_Diff-2x2-binning": 0.709,
    "4D_Diff-4x4-binning": 0.609,
}


def shannon_entropy_limit(sparsity):
    """Calculate sparsity-only (binary entropy) compression upper bound.

    This uses only the zero fraction (sparsity) and ignores the entropy carried by
    non-zero values, so it is an optimistic upper bound.
    """
    p0 = np.clip(sparsity, 1e-10, 1 - 1e-10)
    H2 = -p0 * np.log2(p0) - (1 - p0) * np.log2(1 - p0)
    return 16 / H2


def main():
    # Load aggregated statistics
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    stats_file = repo_root / "results" / "aggregated" / "statistics.csv"
    df = pd.read_csv(stats_file)

    # Get best compression for each dataset
    # uint16 datasets only -- see module docstring
    datasets = [
        "3D_EELS",
        "4D_EELS",
        "4D_Diff",
    ]
    sparsity = []
    compression = []

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        best_compression = dataset_df["compression_ratio_mean"].max()
        compression.append(best_compression)
        sparsity.append(DATASET_SPARSITY[dataset])
        print(
            f"{dataset}: sparsity={DATASET_SPARSITY[dataset]:.3f}, compression={best_compression:.2f}×"
        )


    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Shannon entropy limit
    s_theory = np.linspace(0.4, 0.95, 100)
    c_theory = shannon_entropy_limit(s_theory)
    shannon_line = ax.plot(
        s_theory * 100, c_theory, "k--", linewidth=3, alpha=0.5, zorder=1
    )[0]

    # Plot data points.
    # Colours must match Figure 1, which assigns viridis(0.2..0.9) across ALL FIVE
    # datasets ordered by descending sparsity. We select the entries for the three
    # datasets shown here so a dataset keeps its colour between figures.
    _all_by_sparsity = sorted(DATASET_SPARSITY, key=DATASET_SPARSITY.get, reverse=True)
    _ramp = plt.cm.viridis(np.linspace(0.2, 0.9, len(_all_by_sparsity)))
    _colour_of = dict(zip(_all_by_sparsity, _ramp))
    colors = [_colour_of[d] for d in datasets]
    legend_labels = [
        "3D EELS",
        "4D EELS",
        "4D Diff.",
    ]

    for i, (s_i, c_i, color, label) in enumerate(
        zip(sparsity, compression, colors, legend_labels)
    ):
        ax.scatter(
            s_i * 100,
            c_i,
            s=400,
            c=[color],
            edgecolors="black",
            linewidth=2.5,
            alpha=0.8,
            zorder=3,
            label=label,
        )

    # Formatting
    ax.set_xlabel("Sparsity (%)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Compression Ratio", fontsize=22, fontweight="bold")
    ax.set_title(
        "Compression Ratio vs Data Sparsity\n(Best Implementation per Dataset)",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.5)
    ax.set_xlim(45, 95)
    ax.set_ylim(0, max(compression) * 1.1)

    # Legend 1 (upper left): sparsity-only upper bound
    legend1 = ax.legend(
        [shannon_line],
        ["Binary-entropy upper bound (sparsity-only)"],
        loc="upper left",
        fontsize=16,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )
    ax.add_artist(legend1)

    # Legend 2 (lower right): Dataset labels
    handles, labels = ax.get_legend_handles_labels()
    dataset_handles = handles[-len(datasets) :]
    dataset_labels = labels[-len(datasets) :]
    ax.legend(
        dataset_handles,
        dataset_labels,
        loc="lower right",
        fontsize=16,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        title="Datasets",
        title_fontsize=18,
    )

    # Save figure
    plt.tight_layout()

    output_dir = repo_root / "paper" / "generated" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / "figure_3.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"\n✓ Saved: {output_pdf}")

    print("\n✓ Figure 3 regenerated successfully!")


if __name__ == "__main__":
    main()
