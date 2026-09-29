#!/usr/bin/env python3
"""Check that a regenerated dataset is bit-for-bit the published one.

The Supporting Information claims the simulation "reproduces the published
cube bit-for-bit; the counts, the enlarged lambda patterns, the virtual images
and the atomic coordinates all match by SHA-256". This is what performs that
check, so the claim is one a reader can settle rather than one they must take
on trust.

Why hash the arrays rather than the ``.npz`` file: the container records a
timestamp and compresses its members, so two archives holding identical arrays
differ as files. Hashing each array's own bytes compares the physics and not
the packaging.

Usage
-----
Re-run the simulation, then::

    uv run python -m paper_artifacts.verify_reproduction \\
        /path/to/wse2_pristine_128x128_374e.npz

Exit status is 0 if every array matches and 1 otherwise, so it drops into a
test suite or a CI step unchanged. With no path it reads ``FIGURE_DATA_DIR``.

It lives beside the ``simulation`` subpackage rather than inside it so that it
imports with numpy alone: checking a published dataset should not require the
multislice stack, and ``simulation/__init__`` applies the PySlice patches on
import.

Regenerating the reference digests is deliberate and rare -- only when the
dataset is reissued on purpose::

    uv run python -m paper_artifacts.verify_reproduction --write <npz>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

DIGESTS = Path(__file__).with_name("digests.json")
DEFAULT_NPZ = "wse2_pristine_128x128_374e.npz"

# The four things the Supporting Information names, spelled out as the arrays
# that carry them. Order is display order, not significance.
ARRAYS = (
    "counts",           # the datacube itself
    "pattern_lambda",   # the three enlarged lambda patterns of Figure 6
    "pattern_counts",   # and the Poisson draw from them
    "adf_from_lambda",  # the virtual images, noiseless and sampled
    "adf_from_counts",
    "bf_from_lambda",
    "bf_from_counts",
    "atom_positions",   # the specimen
    "atom_species",
)


def digest(array) -> str:
    """SHA-256 over an array's raw bytes, made C-contiguous first.

    ``np.ascontiguousarray`` matters: a transposed or sliced view has the same
    values in a different memory order and would hash differently for reasons
    that have nothing to do with the simulation.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def digests_of(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=True) as data:
        missing = [k for k in ARRAYS if k not in data.files]
        if missing:
            raise KeyError(f"{path.name} is missing {', '.join(missing)}")
        return {k: digest(data[k]) for k in ARRAYS}


def resolve(argpath: str | None) -> Path:
    if argpath:
        return Path(argpath)
    root = os.environ.get("FIGURE_DATA_DIR")
    if not root:
        sys.exit("give a path to the .npz, or set FIGURE_DATA_DIR")
    return Path(root) / DEFAULT_NPZ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="?", help="dataset to check")
    ap.add_argument("--write", action="store_true",
                    help="overwrite the reference digests from this dataset")
    args = ap.parse_args()

    path = resolve(args.npz)
    if not path.exists():
        sys.exit(f"no such file: {path}")
    try:
        found = digests_of(path)
    except KeyError as exc:
        sys.exit(str(exc).strip("'"))

    if args.write:
        DIGESTS.write_text(json.dumps({
            "comment": ("SHA-256 over each array's raw bytes, C-contiguous. "
                        "Written by verify_reproduction.py --write; regenerate "
                        "only when the dataset is deliberately reissued."),
            "dataset": path.name,
            "arrays": found,
        }, indent=2) + "\n")
        print(f"wrote {DIGESTS}")
        return 0

    if not DIGESTS.exists():
        sys.exit(f"no reference digests at {DIGESTS}; run with --write first")
    expected = json.loads(DIGESTS.read_text())["arrays"]

    width = max(len(k) for k in ARRAYS)
    bad = []
    for key in ARRAYS:
        want, got = expected.get(key), found[key]
        ok = want == got
        if not ok:
            bad.append(key)
        print(f"  {key:<{width}}  {'OK  ' if ok else 'FAIL'}  {got[:16]}"
              + ("" if ok else f"  expected {want[:16] if want else '(absent)'}"))

    print()
    if bad:
        print(f"{len(bad)} of {len(ARRAYS)} arrays differ: {', '.join(bad)}")
        print("The regenerated dataset is not the published one.")
        return 1
    print(f"all {len(ARRAYS)} arrays match; this is the published dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
