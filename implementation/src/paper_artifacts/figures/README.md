# Conceptual figure sources

Scripts for the two conceptual figures of the Discussion.

| script | manuscript purpose |
|---|---|
| `panel_inference_modes.py` | Peirce's three modes — abduction, induction, deduction. Distinguishes "inference" in the sense the manuscript uses from a forward pass through a model. |
| `event_detection.py` | Makes interpretive reduction concrete: the analog trace is discarded and a claim about the event is stored. |

Both are self-contained (numpy + matplotlib only) and write `.png` and `.pdf`
next to themselves.

Regenerate:

    cd implementation/src/paper_artifacts/figures
    uv run python panel_inference_modes.py
    uv run python event_detection.py
