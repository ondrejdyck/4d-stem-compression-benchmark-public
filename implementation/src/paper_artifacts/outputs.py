"""Where generated artifacts go, and the one function that puts them there.

Every figure in the manuscript is ``paper/generated/figures/figure_N.pdf``,
written directly by its generator. There is no copy step, and no generator
chooses its own location or filename.

The numbering lives on the output rather than on the script. A figure's number
is a fact about the manuscript and belongs on the file the manuscript includes;
a script's name is a fact about the code and should survive the paper
reordering its figures.

PDF is the artifact, because that is what the manuscript includes and what
journals accept. PNG and SVG are written only on request -- they are for talks
and for opening in a vector editor, not for the paper.
"""

from __future__ import annotations

import os
from pathlib import Path

# outputs.py -> paper_artifacts -> src -> implementation -> repository root
REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO / "paper" / "generated" / "figures"


def preview_requested() -> bool:
    """Whether to also write PNG and SVG.

    Scripts with an argument parser pass ``--preview`` through explicitly. The
    three that run as flat module-level scripts have no parser, so they read
    FIGURE_PREVIEW instead; setting it to anything but empty or 0 turns the
    extra formats on for the whole run.
    """
    return os.environ.get("FIGURE_PREVIEW", "").strip() not in ("", "0")


def save_figure(fig, number: int, preview: bool = False, **savefig_kwargs) -> Path:
    """Write ``figure_<number>.pdf``, and the preview formats if asked.

    ``fig`` may be a Figure or a pyplot module -- both expose ``savefig`` with
    the same signature, which lets the call sites stay as they were written.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", **savefig_kwargs}

    pdf = FIGURE_DIR / f"figure_{number}.pdf"
    fig.savefig(pdf, **kwargs)
    print(f"wrote {pdf.relative_to(REPO)}")

    if preview:
        for suffix, extra in ((".png", {"dpi": 300}), (".svg", {})):
            path = pdf.with_suffix(suffix)
            fig.savefig(path, **{**kwargs, **extra})
            print(f"wrote {path.relative_to(REPO)}")
    return pdf


SIMULATED_DATASET = "wse2_pristine_128x128_374e.npz"


def simulated_dataset_dir() -> Path:
    """Where the simulated dataset lives: FIGURE_DATA_DIR, or ~/4dstem-figure-data.

    The cube is a few hundred megabytes and is not committed, so it sits
    outside the repository and the generators are told where by environment.
    Defined once here because four callers need the same answer and a default
    that drifted between them would be found only by a reader whose
    regenerated figure disagreed with the published one.
    """
    root = os.environ.get("FIGURE_DATA_DIR")
    return Path(root) if root else Path.home() / "4dstem-figure-data"


def simulated_dataset_path() -> Path:
    return simulated_dataset_dir() / SIMULATED_DATASET
