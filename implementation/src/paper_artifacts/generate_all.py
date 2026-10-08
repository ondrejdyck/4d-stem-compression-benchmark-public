#!/usr/bin/env python3
"""Regenerate every figure and table in the manuscript.

One entry point for the whole artifact set, so that a change to the benchmark
results or to a generator can be propagated in a single command rather than by
remembering twelve of them.

Figure 6 and Table 6 rest on the simulated dataset, which is not committed --
it is several hundred megabytes and regenerable exactly. If it is absent, those
two are skipped with an explanation rather than failing the run; everything
else comes from the committed statistics in ``results/``. (Figure 7 needs no
data at all, despite also belonging to the Discussion.)

Usage
-----
    cd implementation/src
    uv run python -m paper_artifacts.generate_all
    uv run python -m paper_artifacts.generate_all --only figures
    uv run python -m paper_artifacts.generate_all --list

Exit status is 0 if every artifact that could be built was built, and 1 if any
generator failed. A skipped artifact is not a failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from paper_artifacts.outputs import simulated_dataset_path

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]

# (artifact, how to invoke, needs the simulated dataset)
#
# Everything runs as -m from implementation/src, which is what makes
# paper_artifacts importable given package = false in the root pyproject.
# The "kind" column is kept so a future plain script can be added without
# changing the runner.
FIGURES = [
    ("Figure 1  cross-dataset performance", "module", "paper_artifacts.figures.combined_performance", False),
    ("Figure 2  multi-dimensional comparison", "module", "paper_artifacts.figures.radar_chart", False),
    ("Figure 3  chunking strategy", "module", "paper_artifacts.figures.chunking_comparison", False),
    ("Figure 4  sparsity against compression", "module", "paper_artifacts.figures.sparsity_compression", False),
    ("Figure 5  modes of inference", "module", "paper_artifacts.figures.panel_inference_modes", False),
    ("Figure 6  the simulated dataset", "module", "paper_artifacts.figures.simulated_dataset", True),
    ("Figure 7  event-driven detection", "module", "paper_artifacts.figures.event_detection", False),
]

TABLES = [
    ("Table 1  datasets", "module", "paper_artifacts.tables.tab_methods_datasets", False),
    ("Table 3  dataset characteristics", "module", "paper_artifacts.tables.tab_dataset_summary", False),
    ("Table 4  implementation families", "module", "paper_artifacts.tables.tab_implementation_families", False),
    ("Table 5  chunking strategy", "module", "paper_artifacts.tables.tab_chunking_summary", False),
    ("Table 6  cost of storing the simulated cube", "module", "paper_artifacts.tables.tab_generative_codelength", True),
]

# Table 2 is written directly in the manuscript and has no generator.


def run(kind: str, target: str) -> tuple[bool, str]:
    cmd = [sys.executable, target] if kind == "script" else [sys.executable, "-m", target]
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
    return False, tail[-1] if tail else f"exit {proc.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=("figures", "tables"),
                    help="build just one kind (default: both)")
    ap.add_argument("--list", action="store_true",
                    help="list the artifacts and their generators, build nothing")
    args = ap.parse_args()

    wanted = {"figures": FIGURES, "tables": TABLES}.get(
        args.only, FIGURES + TABLES)

    if args.list:
        for name, kind, target, _ in wanted:
            print(f"  {name:<46} {target}")
        return 0

    npz = simulated_dataset_path()
    have_sim = npz.exists()
    if not have_sim:
        print(f"Simulated dataset not found at {npz}")
        print("  Figure 6 and Table 6 will be skipped. To build them, generate it")
        print("  first -- see implementation/src/paper_artifacts/simulation/README.md.")
        print("  Set FIGURE_DATA_DIR if it lives somewhere other than the default.\n")

    failed, built, skipped = [], 0, 0
    width = max(len(n) for n, _, _, _ in wanted)
    for name, kind, target, needs_sim in wanted:
        if needs_sim and not have_sim:
            print(f"  {name:<{width}}  skipped (needs the simulated dataset)")
            skipped += 1
            continue
        print(f"  {name:<{width}}  ", end="", flush=True)
        t0 = time.monotonic()
        ok, why = run(kind, target)
        if ok:
            print(f"ok  {time.monotonic() - t0:5.1f}s")
            built += 1
        else:
            print(f"FAILED  {why[:90]}")
            failed.append(name)

    print(f"\n{built} built, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
