# Multi-Run Benchmark System

This system runs compression benchmarks multiple times to measure performance variability and compute statistics.

## Quick Start

### Weekend Run (10 iterations)

```bash
cd implementation/src
.venv/bin/python run_multiple_benchmarks.py --n-runs 10
```

This will:
- Run the full benchmark suite 10 times
- Take approximately 40-50 hours total
- Create directories: `results/run_001/`, `run_002/`, ..., `run_010/`
- Each run is independent and complete

### Monday Analysis

```bash
cd implementation/src
.venv/bin/python aggregate_multi_run_results.py
```

This will:
- Combine all 10 runs
- Compute mean, std, min, max, median, CV% for each metric
- Save to `results/aggregated/statistics.csv`
- Generate summary report with variability analysis

## Directory Structure

```
results/
├── run_001_20241205_143022/          # First run (timestamped)
│   ├── 4D_EELS/
│   │   ├── benchmark_results.csv
│   │   ├── metadata.json
│   │   └── ...
│   ├── 4D_Diff/
│   └── ...
├── run_002_20241205_150145/          # Second run
│   └── ...
├── run_010_20241205_183456/          # Tenth run
│   └── ...
├── aggregated/                        # Combined analysis (Monday)
│   ├── statistics.csv                 # Mean ± std for all metrics
│   ├── all_runs_combined.csv          # All raw data
│   ├── metadata.json                  # Aggregation info
│   └── summary_report.txt             # Human-readable summary
└── multi_run_summary_20241205_183500.json  # Overall progress log
```

## Key Changes from Single-Run System

### 1. Benchmark Code (`compression_benchmark.py`)
- **Changed:** Read operation now runs **once** (was 3 iterations)
- **Why:** Variability measured across full benchmark runs, not within-run iterations
- **Impact:** ~33% faster per run

### 2. Wrapper Script (`run_multiple_benchmarks.py`)
- Runs `run_all_benchmarks.py` N times
- Creates timestamped directories (no overwrites)
- Tracks progress and handles failures
- Can resume from specific run number

### 3. Aggregation Script (`aggregate_multi_run_results.py`)
- Loads all `run_XXX/` directories
- Computes statistics per dataset/method
- Outputs CSV with `_mean`, `_std`, `_min`, `_max`, `_cv_percent` columns

## Usage Examples

### Run specific datasets only
```bash
.venv/bin/python run_multiple_benchmarks.py --n-runs 10 --datasets 4D_EELS 3D_EELS
```

### Resume from run 6 (if interrupted)
```bash
.venv/bin/python run_multiple_benchmarks.py --n-runs 10 --start-run 6
```

### Run with custom paths
```bash
.venv/bin/python run_multiple_benchmarks.py \
    --n-runs 10 \
    --data-dir /path/to/data \
    --output-base /path/to/results
```

### Aggregate with custom directory
```bash
.venv/bin/python aggregate_multi_run_results.py --results-dir /path/to/results
```

## Expected Runtime

**Per run:** ~4-5 hours (all 5 datasets)
**10 runs:** ~40-50 hours total
**Weekend:** Perfect for Friday evening → Monday morning

## Statistics Computed

For each metric (compression_ratio, write_time, read_time, etc.):
- **Mean:** Average across all runs
- **Std:** Standard deviation
- **Min:** Minimum value observed
- **Max:** Maximum value observed
- **Median:** Middle value
- **CV%:** Coefficient of variation = (std/mean) × 100

## Interpreting Results

### Compression Ratio
- **Expected CV%:** <0.1% (deterministic, should be identical)
- **If CV% > 1%:** Something is wrong (file corruption?)

### Write/Read Times
- **Expected CV%:** 2-10% (system load variability)
- **If CV% > 20%:** High system load or disk contention

### Throughput
- **Expected CV%:** 5-15% (derived from timing)
- Inverse relationship with time variability

## Troubleshooting

### Run failed partway through
```bash
# Check which run failed
cat results/multi_run_summary_*.json

# Resume from next run
.venv/bin/python run_multiple_benchmarks.py --n-runs 10 --start-run 7
```

### Disk space issues
Each run creates ~8-10 GB of temporary files (deleted after each method).
Final results: ~50 MB per dataset × 5 datasets × 10 runs = ~2.5 GB total.

### Aggregation fails
```bash
# Check if all runs completed
ls -la results/run_*/*/benchmark_results.csv | wc -l

# Should be: 5 datasets × 10 runs = 50 files
```

## Next Steps After Aggregation

1. **Update visualization scripts** to use `statistics.csv`
2. **Add error bars** to all figures (use `_min` and `_max` columns)
3. **Update paper** with statistical analysis
4. **Report CV%** for key metrics in paper text

## Files Modified

- `compression_benchmark.py` - Single read operation
- `run_multiple_benchmarks.py` - NEW wrapper script
- `aggregate_multi_run_results.py` - NEW aggregation script
- Visualization scripts - TODO: add error bar support

## Backward Compatibility

The system is backward compatible:
- Old single-run results still work
- Visualization scripts fall back to single-run mode if no `aggregated/` directory exists
- Can mix old and new results (aggregation script only looks for `run_XXX/` directories)
