# 4D STEM Data Compression Benchmark

**Systematic evaluation of compression algorithms for four-dimensional scanning transmission electron microscopy (4D STEM) datasets**

## Overview

This repository contains the implementation code for benchmarking various compression strategies on 4D STEM data. The goal is to identify optimal compression algorithms that balance file size reduction, read/write performance, and ease of implementation for interactive visualization applications.

## Project Structure

```
implementation/
├── src/                    # Source code
│   └── compression_benchmark.py  # Main benchmark script
├── notebooks/              # Jupyter notebooks for interactive analysis
│   └── run_benchmark.ipynb       # Main benchmark workflow
├── tests/                  # Unit tests
├── results/                # Benchmark results (organized by dataset)
│   ├── 64x64_Test/
│   │   ├── compression_analysis_summary.md
│   │   ├── detailed_results.txt
│   │   └── compression_benchmark.png
│   └── [other datasets]/
├── data/                   # Dataset inventory and metadata
├── docs/                   # Additional documentation
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

## Key Findings (Preliminary)

- **Best compression ratio**: 28.9x with HDF5 gzip level 9
- **Best balance**: 27.1x with HDF5 gzip level 6 (18s write, 0.8s read)
- **Fastest**: 14.4x with HDF5 LZF (6s write, 1.5s read)
- **Data sparsity**: 92.8% zeros in typical 4D STEM diffraction patterns

## Algorithms Tested

### Currently Implemented
- **HDF5 compression**: gzip (levels 1, 6, 9), LZF, szip
- **Sparse storage**: CSR matrix format
- **Custom strategies**: uint8 + overflow map, simple gzip

### Planned Additions
- Blosc (fast scientific data compression)
- Zstandard (modern general-purpose)
- LZ4 (extremely fast decompression)
- Bitshuffle + LZ4 (bit-reordering for low-bit-depth data)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/4d-stem-compression-benchmark.git
cd 4d-stem-compression-benchmark/implementation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Recommended: Jupyter Notebook Workflow

The easiest way to run benchmarks is using the provided Jupyter notebook:

```bash
cd implementation/notebooks
jupyter notebook run_benchmark.ipynb
```

The notebook provides:
- Interactive data loading (MIB or HDF5)
- Easy dataset switching
- Custom visualization and analysis
- Export to LaTeX tables for publication

### Python Script (Command Line)

For automated benchmarking:

```bash
cd implementation
python src/compression_benchmark.py
```

This runs the benchmark on the default dataset (64×64 Test.mib).

### Python API (Custom Integration)

For integration into your own analysis pipeline:

```python
from src.compression_benchmark import run_benchmark
from pathlib import Path
import numpy as np

# Load your 4D STEM data (shape: scan_y, scan_x, det_y, det_x)
data_4d = load_your_data('path/to/dataset')

# Run comprehensive benchmark
output_dir = Path('results/my_dataset')
results = run_benchmark(
    data_4d=data_4d,
    output_dir=output_dir,
    dataset_name='my_dataset'
)

# Access results
print(f"Best compression: {results['sorted_results'][0]}")
print(f"Sparsity: {results['sparsity_info']['sparsity']*100:.1f}%")
```

### Individual Benchmark Functions

For fine-grained control:

```python
from src.compression_benchmark import (
    analyze_data_sparsity,
    benchmark_hdf5_compression,
    benchmark_sparse_storage,
    benchmark_custom_compression
)

# Run specific benchmarks
sparsity_info = analyze_data_sparsity(data_4d)
hdf5_results = benchmark_hdf5_compression(data_4d, output_dir)
sparse_results = benchmark_sparse_storage(data_4d, output_dir)
custom_results = benchmark_custom_compression(data_4d, output_dir)
```

## Dataset Requirements

The benchmark expects 4D numpy arrays with shape `(scan_y, scan_x, detector_y, detector_x)`:
- **Scan dimensions**: Typically 64×64 to 512×512
- **Detector dimensions**: Typically 256×256 to 2048×2048
- **Data type**: uint16 (12-bit or 16-bit detector data)
- **File format**: MIB, HDF5, or numpy-compatible formats

## Benchmark Metrics

For each compression method, we measure:
- **Compression ratio**: Original size / compressed size
- **Write time**: Time to compress and save data
- **Read time**: Time to load and decompress sample chunks
- **File size**: Actual disk space used
- **Access patterns**: Performance for different slicing operations

## Results Interpretation

### Compression Ratio
- **>25x**: Excellent (typical for sparse 4D STEM data)
- **15-25x**: Good
- **10-15x**: Moderate
- **<10x**: Poor (consider alternative strategies)

### Write/Read Times
- **Write**: One-time preprocessing cost (acceptable up to minutes)
- **Read**: Critical for interactive use (target <1 second)

## Contributing

This is research code intended for publication. Contributions should focus on:
1. Adding new compression algorithms
2. Testing on diverse datasets
3. Improving benchmark accuracy and reproducibility
4. Documenting edge cases and failure modes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dyck2025compression,
  title={Compression Strategies for Four-Dimensional STEM Data},
  author={Dyck, Ondrej and others},
  journal={TBD},
  year={2025}
}
```

## License

MIT License (or specify your preferred license)

## Contact

Ondrej Dyck - ondrej.dyck@gmail.com

## Acknowledgments

- Andy Lupini for suggesting this analysis
- ORNL for computational resources
- Quantum Detectors for Merlin detector data
