#!/usr/bin/env python3
"""Generate the generative-code-length table for the Discussion.

What it compares, for the simulated 4D-STEM datacube of Figure 6:

- what it costs to store, raw and under two general-purpose lossless codecs;
- what it costs to store under the shortest code available to something that
  knows the detector exactly and the specimen not at all (the Poisson term,
  computed during the simulation and carried in the dataset's metadata);
- what it costs to store the ADF image the cube is usually reduced to;
- how many bytes were used to generate the cube in the first place.

The last line is the point of the table and is only available because the data
are simulated. It is an *upper* bound on the generating description: the
parameters are serialised as minified JSON, keys and decimal digits and all,
with no attempt to pack them. The conservative direction is the one that
weakens our claim, so no packing is attempted.

Inputs (source of truth):
- the simulated dataset written by paper_artifacts.simulation.simulate_dataset

Outputs:
- LaTeX:  paper/generated/tables/table_generative_codelength.tex
- ASCII:  paper/generated/tables_ascii/table_generative_codelength.txt
- CSV:    paper/generated/tables_csv/table_generative_codelength.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DEFAULT_NPZ = Path(
    os.environ.get("FIGURE_DATA_DIR", Path.home() / "4dstem-figure-data")
) / "wse2_pristine_128x128_374e.npz"


# --------------------------------------------------------------------------
# The generating description.
#
# Included: everything the simulation consumes as an input.
# Excluded as DERIVED, because the simulation computes them from the above:
#   wavelength (from the voltage), probe diameter, slice thickness
#   (box_z / n_slices), real-space sampling (box / real_grid), scan step
#   (box / scan_shape), dose (beam current x dwell / e).
# Excluded as DOWNSTREAM: the ADF inner and outer angles. They define a
#   virtual detector applied *to* the cube, not anything needed to produce it.
# Excluded as CARRYING NO INFORMATION ABOUT THE SPECIMEN: the two random
#   seeds. Sixteen further bytes make the cube bit-for-bit reproducible, which
#   is a fact about this file's provenance rather than about the world; see
#   the discussion in the manuscript.
# --------------------------------------------------------------------------

def generating_description(meta: dict) -> tuple[dict, dict]:
    """Return the (specimen, instrument) parameter sets, drawn from metadata."""
    build = meta["build_record"]
    specimen = {
        "compound": "WSe2",
        "polytype": meta["polytype"],
        "a_A": meta["lattice_a_A"],
        "se_se_A": meta["se_se_thickness_A"],
        "layers": build["layers"],
        "supercell": build["orthogonal_supercell"],
        "repeats": build["lateral_repeats"],
        "vacuum_per_side_A": meta["vacuum_per_side_A"],
    }
    instrument = {
        "voltage_V": meta["voltage_V"],
        "alpha_rad": meta["convergence_semiangle_rad"],
        "theta_max_rad": meta["band_limit_rad"],
        "current_A": meta["beam_current_A"],
        "dwell_s": meta["dwell_s"],
        "phonon_sigma_A": meta["frozen_phonon_sigma_A"],
        "phonon_configs": meta["frozen_phonon_configs"],
        "real_grid": meta["real_grid"],
        "n_slices": meta["n_slices"],
        "scan_shape": meta["scan_shape"],
    }
    return specimen, instrument


def minified(d: dict) -> bytes:
    return json.dumps(d, separators=(",", ":"), sort_keys=True).encode()


def compute_rows(npz_path: Path) -> tuple[list[dict], dict]:
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["metadata_json"]))
    counts = d["counts"]
    raw = counts.nbytes

    rows: list[dict] = []

    def add(label, nbytes, recovers, asserts):
        rows.append({"label": label, "bytes": int(round(nbytes)),
                     "ratio": raw / nbytes, "recovers": recovers,
                     "asserts": asserts})

    add("Raw datacube (uint8)", raw, "---", "---")
    # The one off-the-shelf reference, configured exactly as the benchmark
    # above configured it: hdf5plugin Blosc with the zstd codec at clevel 3
    # and byte shuffling, on balanced chunks, measured as the size of the HDF5
    # file on disk. Compressing a flat buffer with a standalone codec would
    # not be comparable to any ratio reported earlier in the paper.
    import os
    import tempfile

    import h5py
    import hdf5plugin

    cube4d = counts.reshape(*meta["scan_shape"], *counts.shape[1:])
    sy, sx, qy, qx = cube4d.shape
    chunks = (min(16, sy), min(16, sx), min(128, qy), min(128, qx))
    with tempfile.TemporaryDirectory() as tmp:
        h5 = os.path.join(tmp, "cube.h5")
        with h5py.File(h5, "w") as fh:
            fh.create_dataset(
                "data", data=cube4d, chunks=chunks,
                **hdf5plugin.Blosc(cname="zstd", clevel=3,
                                   shuffle=hdf5plugin.Blosc.SHUFFLE))
        add("Blosc Zstd, balanced chunks", os.path.getsize(h5),
            "exactly", "nothing")

    specimen, instrument = generating_description(meta)
    model_bytes = len(minified(specimen)) + len(minified(instrument))

    # The generating parameters give lambda; an arithmetic coder holding lambda
    # then codes the counts in the length recorded during the simulation. The
    # pair is a complete, self-contained, *lossless* representation of the cube:
    # a coder with a wrong probability model emits a longer code, never a wrong
    # one, so nothing here is asserted. Not a codec anyone can ship -- decoding
    # means re-running the multislice, and lambda is only constructible because
    # the specimen is already known -- but it bounds what a model of the source
    # is worth without discarding anything.
    add("Model-based code, counts retained",
        model_bytes + meta["coded_bits_this_draw"] / 8, "exactly", "nothing")

    adf = d["adf_from_lambda"].astype(np.float32)
    add("ADF image (float32)", adf.nbytes, "not at all", "nothing")

    # The same 347 bytes with the residual thrown away. One line lower in the
    # table, and the only row that can be false.
    add("Model-based reduction, counts discarded", model_bytes, "not at all", "the specimen")

    rows.sort(key=lambda r: -r["bytes"])

    extra = {
        "meta": meta,
        "specimen": specimen,
        "instrument": instrument,
        "specimen_bytes": len(minified(specimen)),
        "instrument_bytes": len(minified(instrument)),
        "coded_bits_this_draw": meta["coded_bits_this_draw"],
        "ideal_code_bits": meta["ideal_code_bits"],
    }
    return rows, extra


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:,.1f}~MiB"
    if n >= 1024:
        return f"{n / 1024:,.1f}~KiB"
    return f"{n:,}~B"


def fmt_ratio(r: float) -> str:
    if r < 10:
        return f"{r:.1f}$\\times$"
    if r < 100_000:
        return f"{r:,.0f}$\\times$"
    return f"{r:,.0f}$\\times$"


def write_latex(rows, extra, path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{What it costs to store the simulated datacube of "
        r"Figure~\ref{fig:4d_dataset}, against what it cost to generate it. A "
        r"representation that makes an assertion about the world can be false. The model-based "
        r"reduction makes a claim about the specimen rather than the data; the other "
        r"rows report data-derived quantities. "
        r"Section~\ref{sec:coarsening} develops the distinction. How each "
        r"entry was computed is given in Supporting Information Section~S2.}",
        r"\label{tab:generative}",
        # five columns run a hair past \textwidth at the default column
        # separation; 4pt is invisible and leaves margin.
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrll}",
        r"\hline",
        r"Representation & Size & Reduction & Recovers & Asserts \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {fmt_bytes(r['bytes'])} & "
            f"{'---' if r['ratio'] == 1 else fmt_ratio(r['ratio'])} & "
            f"{r['recovers']} & {r['asserts']} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_ascii(rows, extra, path: Path) -> None:
    w = max(len(r["label"]) for r in rows) + 2
    out = [f"{'Representation':<{w}}{'Bytes':>16}{'Reduction':>12}"
           f"  {'Recovers':<10}{'Asserts':<14}", "-" * (w + 54)]
    for r in rows:
        ratio = "---" if r["ratio"] == 1 else f"{r['ratio']:,.0f}x"
        out.append(f"{r['label']:<{w}}{r['bytes']:>16,}{ratio:>12}"
                   f"  {r['recovers']:<10}{r['asserts']:<14}")
    out += ["",
            f"specimen parameters:   {extra['specimen_bytes']} B  "
            f"{json.dumps(extra['specimen'], separators=(',', ':'), sort_keys=True)}",
            f"instrument parameters: {extra['instrument_bytes']} B  "
            f"{json.dumps(extra['instrument'], separators=(',', ':'), sort_keys=True)}",
            "",
            f"ideal code length, expected : {extra['ideal_code_bits'] / 8:,.0f} B",
            f"ideal code length, this draw: {extra['coded_bits_this_draw'] / 8:,.0f} B",
            f"  agreement: {abs(extra['coded_bits_this_draw'] - extra['ideal_code_bits']) / extra['ideal_code_bits']:.2e}",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


def write_csv_out(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["label", "bytes", "ratio",
                                    "recovers", "asserts"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main() -> None:
    npz = Path(os.environ.get("FIGURE_DATASET", DEFAULT_NPZ))
    print(f"reading {npz}")
    rows, extra = compute_rows(npz)
    write_latex(rows, extra, REPO / "paper/generated/tables/table_generative_codelength.tex")
    write_ascii(rows, extra, REPO / "paper/generated/tables_ascii/table_generative_codelength.txt")
    write_csv_out(rows, REPO / "paper/generated/tables_csv/table_generative_codelength.csv")
    print("\nwrote paper/generated/tables{,_ascii,_csv}/table_generative_codelength.*")


if __name__ == "__main__":
    main()
