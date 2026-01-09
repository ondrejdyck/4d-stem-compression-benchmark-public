# Benchmark Workflow

**Updated**: 2025-11-06  
**New workflow**: Separate benchmarking from visualization for faster iteration

---

## Overview

The benchmark workflow is now split into two phases:

1. **Benchmarking** (slow, run once per dataset): Compress data and save results to CSV
2. **Visualization** (fast, iterate freely): Load CSV and create plots

This separation allows you to:
- Run expensive benchmarks once
- Iterate on plots without re-computing
- Easily compare multiple datasets
- Export results in standard formats (CSV, JSON)

---

## Quick Start

### 1. Run Benchmark (Command Line)

```bash
cd implementation

# Benchmark a dataset
.venv/bin/python src/run_benchmark.py data/64x64_Test.emd

# Or with custom output location and name
.venv/bin/python src/run_benchmark.py data/1_256x256_2msec_graphene.emd \
    --output results \
    --name graphene_256
```

**Output:**
```
results/
└── 64x64_Test/
    ├── benchmark_results.csv          # Main results (load in notebook)
    ├── metadata.json                  # Dataset info
    └── 64x64_Test_detailed_results.txt  # Human-readable summary
```

### 2. Visualize Results (Notebook)

```bash
cd implementation/notebooks
jupyter notebook visualize_results.ipynb
```

The notebook will:
- Auto-discover available datasets
- Load CSV files into pandas DataFrames
- Create customizable plots
- Export LaTeX tables for paper

---

## Detailed Workflow

### Phase 1: Benchmarking

#### Option A: Command Line (Recommended)

```bash
# Run benchmark on a dataset
.venv/bin/python src/run_benchmark.py data/dataset.emd

# Options:
#   --output DIR    Output directory (default: ../results)
#   --name NAME     Dataset name (default: filename)
#   --plots         Create summary plots (optional)
```

**What it does:**
1. Loads EMD/HDF5 file
2. Runs all compression algorithms (13 methods × 3 chunk sizes = 39+ tests)
3. Saves results to CSV
4. Saves metadata to JSON
5. Prints summary to console

**Time:** 20-60 minutes depending on dataset size

#### Option B: Python API

```python
from compression_benchmark import load_emd, run_benchmark

# Load data
data_4d = load_emd('data/dataset.emd')

# Run benchmark
results = run_benchmark(
    data_4d, 
    output_dir='results/my_dataset',
    dataset_name='my_dataset',
    save_csv=True,      # Save to CSV
    create_plots=False  # Skip plots (do in notebook)
)
```

### Phase 2: Visualization

Open `notebooks/visualize_results.ipynb` and run cells:

1. **Discover datasets**: Auto-find all benchmark results
2. **Load data**: Read CSV into pandas DataFrame
3. **Basic stats**: Top 10 methods, summary statistics
4. **Plot compression ratios**: Horizontal bar chart with color coding
5. **Plot speed trade-offs**: Scatter plot of compression vs time
6. **Compare datasets**: Side-by-side comparison (if multiple datasets)
7. **Export LaTeX**: Generate tables for paper

**Iterate freely!** Change plots, colors, filters without re-running benchmarks.

---

## File Structure

```
implementation/
├── src/
│   ├── compression_benchmark.py      # Core benchmark functions
│   └── run_benchmark.py              # CLI script (NEW)
├── notebooks/
│   ├── visualize_results.ipynb       # Visualization only (NEW)
│   └── run_benchmark.ipynb           # Old combined notebook (deprecated)
├── results/
│   ├── 64x64_Test/
│   │   ├── benchmark_results.csv     # Load this in notebook
│   │   ├── metadata.json             # Dataset info
│   │   └── *_detailed_results.txt    # Human-readable
│   └── 256x256_graphene/
│       ├── benchmark_results.csv
│       └── metadata.json
└── data/
    ├── 64x64_Test.emd
    └── 1_256x256_2msec_graphene.emd
```

---

## CSV Format

`benchmark_results.csv` contains one row per compression method:

| Column | Description |
|--------|-------------|
| dataset | Dataset name |
| method | Compression method (e.g., "real_space_gzip_9") |
| compression_ratio | Original size / compressed size |
| file_size_mb | Compressed file size in MB |
| write_time | Time to compress and save (seconds) |
| read_time | Time to decompress sample chunks (seconds) |
| chunk_size | HDF5 chunk dimensions (if applicable) |

**Example:**
```csv
dataset,method,compression_ratio,file_size_mb,write_time,read_time,chunk_size
64x64_Test,real_space_gzip_9,28.9,69.2,45.3,0.82,"(32, 32, 256, 256)"
64x64_Test,real_space_blosc_zstd,30.1,66.5,38.7,0.65,"(32, 32, 256, 256)"
```

---

## Metadata Format

`metadata.json` contains dataset information:

```json
{
  "dataset_name": "64x64_Test",
  "data_shape": [64, 64, 256, 1024],
  "data_dtype": "uint16",
  "original_size_mb": 2048.0,
  "sparsity": 0.993,
  "unique_values": 4096,
  "max_value": 4095,
  "mean_value": 12.3
}
```

---

## Comparing Multiple Datasets

### 1. Run benchmarks on each dataset

```bash
.venv/bin/python src/run_benchmark.py data/64x64_Test.emd
.venv/bin/python src/run_benchmark.py data/1_256x256_2msec_graphene.emd
```

### 2. Load all in notebook

```python
# In visualize_results.ipynb
dfs = []
for dataset_name in datasets:
    df = pd.read_csv(f'../results/{dataset_name}/benchmark_results.csv')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Now analyze across datasets
df_all.groupby('dataset')['compression_ratio'].mean()
```

### 3. Create comparison plots

The notebook includes cells for:
- Side-by-side bar charts
- Scatter plots colored by dataset
- Statistical comparisons

---

## Tips & Tricks

### Faster Iteration

**Don't re-run benchmarks!** Once you have the CSV:
- Tweak plots in notebook
- Try different visualizations
- Filter/group data with pandas
- Export different table formats

### Custom Analysis

```python
# Load results
df = pd.read_csv('../results/64x64_Test/benchmark_results.csv')

# Filter to just Blosc methods
df_blosc = df[df['method'].str.contains('blosc')]

# Find best compression/speed trade-off
df['efficiency'] = df['compression_ratio'] / df['write_time']
best = df.nlargest(5, 'efficiency')
```

### Export for Paper

```python
# Top 10 methods as LaTeX table
top_10 = df.nlargest(10, 'compression_ratio')
# ... generate LaTeX (see notebook)

# Or export to Excel for collaborators
df.to_excel('../results/benchmark_results.xlsx', index=False)
```

---

## Troubleshooting

### "No datasets found"
- Make sure you've run benchmarks first
- Check that CSV files exist in `results/*/benchmark_results.csv`

### "Module not found: pandas"
- Install: `uv pip install pandas`
- Or: `pip install -r requirements.txt`

### Benchmark takes too long
- Normal! 30-60 minutes for large datasets
- Run overnight or during lunch
- Results are saved incrementally (can resume if interrupted)

### Want to add more compression methods
- Edit `compression_benchmark.py`
- Add to `compression_methods` dict
- Re-run benchmark (old results preserved)

---

## Migration from Old Workflow

**Old way** (run_benchmark.ipynb):
- Mixed benchmarking and plotting
- Had to re-run everything to change plots
- Slow iteration

**New way**:
1. Run `python src/run_benchmark.py data/dataset.emd` once
2. Use `visualize_results.ipynb` for all plotting
3. Fast iteration, no redundant computation

**To migrate:**
- Run benchmarks using new CLI
- Use new visualization notebook
- Old notebook still works but is deprecated

---

## Next Steps

1. Run benchmarks on all your datasets
2. Open `visualize_results.ipynb`
3. Create publication-quality figures
4. Export LaTeX tables for paper
5. Iterate on visualization without re-computing!

---

**Questions?** See `README.md` or check the code comments.
