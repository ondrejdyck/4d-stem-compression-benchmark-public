#!/usr/bin/env python3
"""
Regenerate Figure 3 (Sparsity vs Compression) with correct power-law equation
Uses aggregated statistics from 10-run benchmarks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


def shannon_entropy_limit(sparsity):
    """Calculate sparsity-only (binary entropy) compression upper bound.

    This uses only the zero fraction (sparsity) and ignores the entropy carried by
    non-zero values, so it is an optimistic upper bound.
    """
    p0 = np.clip(sparsity, 1e-10, 1 - 1e-10)
    H2 = -p0 * np.log2(p0) - (1 - p0) * np.log2(1 - p0)
    return 16 / H2


def power_law(s, a, b, c):
    """Power law function: C = a * s^b + c"""
    return a * s**b + c


def main():
    # Load aggregated statistics
    # Auto-detect paths relative to this script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    stats_file = repo_root / "results" / "aggregated" / "statistics.csv"
    inv_file = repo_root / "results" / "dataset_inventory.csv"
    df = pd.read_csv(stats_file)
    inv = pd.read_csv(inv_file)[["dataset_id", "sparsity_fraction"]].rename(
        columns={"dataset_id": "dataset", "sparsity_fraction": "sparsity"}
    )

    # Get best compression for each dataset using committed CSV sources
    sparsity = []
    compression = []
    datasets = []

    for _, row in inv.sort_values("sparsity", ascending=True).iterrows():
        dataset = row["dataset"]
        dataset_df = df[df["dataset"] == dataset]
        if dataset_df.empty:
            raise ValueError(f"No aggregated statistics found for dataset={dataset}")
        best_compression = dataset_df["compression_ratio_mean"].max()
        compression.append(best_compression)
        sparsity.append(float(row["sparsity"]))
        datasets.append(dataset)
        print(
            f"{dataset}: sparsity={float(row['sparsity']):.3f}, compression={best_compression:.2f}×"
        )

    # Fit power law
    params, _ = curve_fit(power_law, sparsity, compression, p0=[50, 7, 5])
    a, b, c = params

    # Calculate R²
    residuals = np.array(compression) - power_law(np.array(sparsity), *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(compression) - np.mean(compression)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nPower law fit: C = {a:.1f} × s^{b:.2f} + {c:.1f}")
    print(f"R² = {r_squared:.3f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Shannon entropy limit
    s_theory = np.linspace(0.4, 0.95, 100)
    c_theory = shannon_entropy_limit(s_theory)
    shannon_line = ax.plot(
        s_theory * 100, c_theory, "k--", linewidth=3, alpha=0.5, zorder=1
    )[0]

    # Plot power law fit
    s_fit = np.linspace(min(sparsity), max(sparsity), 100)
    c_fit = power_law(s_fit, *params)
    powerlaw_line = ax.plot(s_fit * 100, c_fit, "r-", linewidth=3.5, zorder=2)[0]

    # Plot data points
    colors = plt.cm.viridis(np.linspace(0, 1, len(sparsity)))
    legend_labels = [
        "3D EELS",
        "4D EELS",
        "4D Diff.",
        "4D Diff. (2×2 bin)",
        "4D Diff. (4×4 bin)",
    ]

    for i, (s, c, color, label) in enumerate(
        zip(sparsity, compression, colors, legend_labels)
    ):
        ax.scatter(
            s * 100,
            c,
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

    # Legend 1 (upper left): sparsity-only upper bound and power-law fit
    legend1 = ax.legend(
        [shannon_line, powerlaw_line],
        [
            "Binary-entropy upper bound (sparsity-only)",
            f"Power Law Fit: $C = {a:.1f} \\cdot s^{{{b:.2f}}} + {c:.1f}$ (R² = {r_squared:.3f})",
        ],
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

    output_pdf = output_dir / "Fig3_sparsity_vs_compression.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"\n✓ Saved: {output_pdf}")

    print("\n✓ Figure 3 regenerated successfully!")


if __name__ == "__main__":
    main()
