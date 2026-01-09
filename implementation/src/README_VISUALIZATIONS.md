# Standalone Visualization Scripts

## Overview

This directory contains standalone Python scripts for generating publication-quality figures. These are easier to work with than Jupyter notebooks for automated figure generation.

All scripts use the shared `data_loader.py` module for consistent data loading and processing.

## Shared Module

### `data_loader.py`
**Purpose:** Shared functions for loading and processing benchmark results

**Key functions:**
- `load_all_benchmarks(results_dir)` - Load all CSVs with metadata
- `calculate_metrics(df)` - Add throughput and algorithm family columns
- `filter_by_chunking(df, type)` - Filter to specific chunking strategy
- `normalize_metrics(df)` - Normalize metrics to 0-1 scale
- `get_best_per_dataset(df)` - Get best method per dataset
- `load_and_process(results_dir, chunking_type, normalize)` - Convenience wrapper

**Usage in scripts:**
```python
from data_loader import load_and_process

# Load data with balanced chunking
df = load_and_process(results_dir, chunking_type='balanced', normalize=False)
```

## Available Scripts

### `plot_radar_chart.py`
**Purpose:** Create radar chart comparing compression algorithms across 3 metrics

**Metrics:**
- Compression ratio
- Write throughput (GB/s)
- Read throughput (GB/s)

**Algorithms compared:**
- Blosc Zstd (best balance)
- Blosc Zlib (high compression)
- Blosc LZ4 (fastest)
- Gzip-9 (traditional high compression)
- Gzip-1 (traditional fast)

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_radar_chart.py
```

**Output:**
- `results/algorithm_radar_chart.png` (480 KB, 300 DPI)
- `results/algorithm_radar_chart.svg` (vector, editable in Inkscape)

**Key findings shown:**
- Blosc LZ4: Largest radar area - best for interactive workflows (7.9× compression, 1.39 GB/s write, 30.2 GB/s read)
- Blosc Zstd: Best balance (13.5× compression, 0.35 GB/s write, 20.2 GB/s read)
- Blosc Zlib: High compression, slower (13.5× compression, 0.13 GB/s write, 7.9 GB/s read)
- Gzip-9: Slow, archival only (12.3× compression, 0.01 GB/s write, 7.9 GB/s read)

---

### `plot_sparsity_compression.py`
**Purpose:** Show relationship between data sparsity and compression ratio

**Features:**
- Scatter plot of 5 datasets with varying sparsity (49.5% to 92.8%)
- Power law fit: C = 50.5 × s^7.11 + 5.1 (R² = 0.989)
- Shannon entropy theoretical limit curve
- Efficiency calculation (achieved vs theoretical limit)

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_sparsity_compression.py
```

**Output:**
- `results/sparsity_vs_compression.png` (317 KB, 300 DPI)
- `results/sparsity_vs_compression.svg` (vector, editable in Inkscape)

**Key findings shown:**
- Sparse data approaches Shannon entropy limit more closely
- 4D EELS (92.8% sparse): 34.9× compression (81.2% of theoretical limit)
- Graphene datasets (60-75% sparse): 7-11× compression (42-61% of limit)
- 3D EELS (49.5% sparse): 4.8× compression (30% of limit)

---

### `plot_speed_tradeoff.py`
**Purpose:** Show tradeoff between compression ratio and read/write speed

**Features:**
- Dual scatter plots (write speed vs compression, read speed vs compression)
- "Ideal region" highlighting for balanced performance
- Use case recommendations for each algorithm

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_speed_tradeoff.py
```

**Output:**
- `results/speed_tradeoff.png` (315 KB, 300 DPI)
- `results/speed_tradeoff.svg` (vector, editable in Inkscape)

**Key findings shown:**
- Blosc LZ4: 17-58× faster write than gzip → interactive analysis
- Blosc Zstd: Best balance → general workflows
- Gzip-9: Very slow (0.01 GB/s write) → not recommended

---

### `plot_cross_dataset_comparison.py`
**Purpose:** Compare compression ratios across multiple datasets

**Features:**
- Grouped horizontal bar chart with intelligent sorting
- Algorithms ordered by mean compression ratio (best at top)
- Datasets ordered by sparsity (most sparse first) within each algorithm
- Background gray bars show mean compression ratio for context
- Shows clear trends in algorithm performance and sparsity effects

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_cross_dataset_comparison.py --top-n 10
```

**Output:**
- `results/cross_dataset_comparison.png` (333 KB, 300 DPI)
- `results/cross_dataset_comparison.svg` (88 KB, vector, editable in Inkscape)

**Key findings shown:**
- Blosc Zstd/Zlib: Best overall compression across all datasets (~13.5× mean)
- Clear sparsity gradient: Higher sparsity → higher compression
- Consistent algorithm ranking across datasets

---

### `plot_cross_dataset_read_speed.py`
**Purpose:** Compare read throughput (GB/s) across multiple datasets

**Features:**
- Grouped horizontal bar chart with intelligent sorting
- Algorithms ordered by mean read throughput (fastest at top)
- Datasets ordered by sparsity (most sparse first)
- Background gray bars show mean read speed for context

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_cross_dataset_read_speed.py --top-n 10
```

**Output:**
- `results/cross_dataset_read_speed.png` (327 KB, 300 DPI)
- `results/cross_dataset_read_speed.svg` (90 KB, vector, editable in Inkscape)

**Key findings shown:**
- Blosc LZ4: Fastest read speed across datasets (~30 GB/s mean)
- "None" (uncompressed): Fast on some datasets but inconsistent
- Gzip variants: Slowest read speeds (2-8 GB/s)

---

### `plot_cross_dataset_write_speed.py`
**Purpose:** Compare write throughput (GB/s) across multiple datasets

**Features:**
- Grouped horizontal bar chart with intelligent sorting
- Algorithms ordered by mean write throughput (fastest at top)
- Datasets ordered by sparsity (most sparse first)
- Background gray bars show mean write speed for context

**Usage:**
```bash
cd implementation/src
../.venv/bin/python plot_cross_dataset_write_speed.py --top-n 10
```

**Output:**
- `results/cross_dataset_write_speed.png` (326 KB, 300 DPI)
- `results/cross_dataset_write_speed.svg` (91 KB, vector, editable in Inkscape)

**Key findings shown:**
- "None" (uncompressed): Fastest write speed (~1.8 GB/s mean)
- Blosc LZ4: Fastest compressed write (~1.4 GB/s mean)
- Blosc Zstd: Moderate write speed (~0.35 GB/s) with high compression

---

### `generate_all_figures.py`
**Purpose:** Master script to generate all publication figures at once

**Features:**
- Runs all 6 visualization scripts in sequence
- Reports success/failure for each script
- Lists generated files with sizes
- Total execution time: ~7 seconds

**Usage:**
```bash
cd implementation/src
../.venv/bin/python generate_all_figures.py
```

**Output:**
- All 12 figure files (6 PNG + 6 SVG)
- Summary report of generation status

## Workflow Advantages

**Standalone scripts vs Jupyter notebooks:**
✓ Easier for Claude to create and modify
✓ Can be run from command line
✓ Reproducible (no cell execution order issues)
✓ Version control friendly
✓ Can be automated in pipelines
✓ Easier to debug

## Future Scripts (Optional)

1. `plot_chunking_analysis.py` - Chunking strategy heatmap (if needed for supplementary materials)
2. `plot_algorithm_families.py` - Algorithm family comparison bars (if needed for discussion)

## Adding New Visualizations

Template for new scripts:
```python
#!/usr/bin/env python3
"""
Script description

Usage:
    python plot_something.py [--results RESULTS_DIR] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import load_and_process  # Use shared module!

def create_plot(df, output_dir):
    """Create the visualization."""
    # Your plotting code here
    pass

def main():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--results', default='../../results')
    parser.add_argument('--output', default='../../results')
    args = parser.parse_args()
    
    # Load data using shared module
    df = load_and_process(Path(args.results), chunking_type='balanced')
    create_plot(df, Path(args.output))

if __name__ == '__main__':
    main()
```

## Notes

- All scripts use the virtual environment: `../.venv/bin/python`
- Default paths assume running from `implementation/src/`
- Output goes to `results/` directory (root level, not implementation/results/)
- Both PNG (raster, 300 DPI) and SVG (vector) formats are generated
- SVG files can be edited in Inkscape while preserving vector quality
- Scripts print summary statistics to console

