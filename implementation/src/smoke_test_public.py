#!/usr/bin/env python3
"""Smoke-test the public benchmark workflow using the included synthetic fixture.

This script validates that the core benchmark engine runs end-to-end without any
private raw datasets.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from compression_benchmark import load_emd, run_benchmark


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "implementation" / "fixtures" / "smoke_test.emd"
    output_dir = repo_root / "results" / "smoke_test_public"

    if not fixture.exists():
        print(f"ERROR: fixture not found: {fixture}")
        return 1

    print("Running public smoke test...")
    data = load_emd(fixture)
    results = run_benchmark(data, output_dir=output_dir, dataset_name="smoke_test")

    expected_csv = output_dir / "benchmark_results.csv"
    expected_meta = output_dir / "metadata.json"
    if not expected_csv.exists() or not expected_meta.exists():
        print("ERROR: expected output files were not created")
        return 1

    print("Smoke test completed successfully.")
    print(f"  dataset shape: {results['data_shape']}")
    print(f"  outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
