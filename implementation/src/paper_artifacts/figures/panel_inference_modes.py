"""
Inference-modes panel -- Peirce's three modes of inference (deduction,
induction, abduction), shown with the classic beans syllogism (each premise
tagged with its f/g/h role) and with Corfield's category-theoretic
triangles, placed to the right of each row.

Run with: python -m paper_artifacts.figures.panel_inference_modes, from implementation/src.
Writes paper/generated/figures/figure_5.pdf.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox

# Embed TrueType rather than matplotlib's default Type 3 fonts. Type 3 is
# rejected by several journals' production systems and carries no ToUnicode
# map, so text in the figure cannot be selected, searched or read aloud.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from paper_artifacts.outputs import save_figure, preview_requested


plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---------------------------------------------------------------------------
# Style constants, shared with event_detection.py.
# ---------------------------------------------------------------------------
SIGNAL_COLOR = "#333333"
SAMPLE_COLOR = "C0"
THRESH_COLOR = "#a33"
TITLE_COLOR = "#555555"
GIVEN_COLOR = "#999999"

# Each syllogism line is (role, text); role "f"/"g"/"h" identifies which
# triangle arrow the statement plays. The role that is *found* (not given)
# in that mode is marked with a leading "∴" and picked out in the mode's
# colour below; the same f/g/h role is always the same statement, in the
# same position, across all three modes -- only which one is concluded
# changes.
RULE = ("g", "All the beans from this bag are white.")
CASE = ("f", "These beans are from this bag.")
RESULT = ("h", "These beans are white.")

MODES = [
    dict(
        name="Deduction", color=SIGNAL_COLOR,
        given="Rule + Case", finds="Result",
        tag="forced\ntruth-preserving",
        f_note=None, g_note=None,
        lines=[RULE, CASE, RESULT], found_role="h",
    ),
    dict(
        name="Induction", color=SAMPLE_COLOR,
        given="Case + Result", finds="Rule",
        tag="ampliative\nfinds the rule",
        f_note=None, g_note=None,
        lines=[CASE, RESULT, RULE], found_role="g",
    ),
    dict(
        name="Abduction", color=THRESH_COLOR,
        given="Rule + Result", finds="Case",
        tag="ampliative\nfinds the case",
        f_note=None, g_note=None,
        lines=[RULE, RESULT, CASE], found_role="f",
    ),
]

# ---------------------------------------------------------------------------
# Sizing. The canvas is built at the width the figure is actually reproduced
# at (\textwidth of the manuscript, 6.5in), so LaTeX includes it at scale 1:1
# and every fontsize below is literally the point size on the printed page.
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 6.5, 3.35

FS_HEADER = 8.0    # column headers
FS_MODE = 9.5      # Deduction / Induction / Abduction
FS_TAG = 6.5       # "ampliative / finds the rule"
FS_GIVEN = 8.5     # "Rule + Case"
FS_FINDS = 8.5     # "Result"
FS_SYL = 8.0       # syllogism lines and their f/g/h tags
FS_VERTEX = 7.5    # A, B, C
FS_VLABEL = 6.5    # "these beans", "bag beans", "white"
FS_EDGE = 6.5      # f, g, h edge labels

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title, subtitle, and footer note are deliberately omitted: in the manuscript
# that material belongs to the LaTeX caption, not to the artwork.


def fx(inches):
    return inches / FIG_W


def fy(inches):
    return inches / FIG_H


# ---------------------------------------------------------------------------
# Column layout, in inches from the lower-left corner. Widths are set by the
# longest string each column has to carry at the sizes above: the syllogism
# ("All the beans from this bag are white." at 8pt) needs ~2.35in and drives
# everything else.
# ---------------------------------------------------------------------------
X_MODE = 0.06
X_GIVEN = 0.92
X_FINDS = 1.88
X_SYL_TAG = 2.44
X_SYL_TXT = 2.72
X_DIAG0 = 5.10
DIAG_W = 1.32

RULE_X0, RULE_X1 = 0.06, 6.44

HEADER_Y = 3.19
ROW_TOP = 2.98
ROW_H = 0.96
APEX_H = 0.34  # triangle height (B above the A-C baseline)

fig.text(fx(X_MODE), fy(HEADER_Y), "Inference", fontsize=FS_HEADER, color=TITLE_COLOR,
          ha="left", fontweight="bold")
fig.text(fx(X_GIVEN), fy(HEADER_Y), "Given", fontsize=FS_HEADER, color=TITLE_COLOR,
          ha="left", fontweight="bold")
fig.text(fx(X_FINDS), fy(HEADER_Y), "Finds", fontsize=FS_HEADER, color=TITLE_COLOR,
          ha="left", fontweight="bold")
fig.text(fx((X_SYL_TXT + X_DIAG0) / 2), fy(HEADER_Y), "The beans syllogism",
          fontsize=FS_HEADER, color=TITLE_COLOR, ha="center", fontweight="bold")
fig.text(fx(X_DIAG0 + DIAG_W / 2), fy(HEADER_Y), "As a triangle",
          fontsize=FS_HEADER, color=TITLE_COLOR, ha="center", fontweight="bold")
fig.add_artist(plt.Line2D([fx(RULE_X0), fx(RULE_X1)], [fy(HEADER_Y - 0.13)] * 2,
                          color="#cccccc", lw=0.8))


def draw_arrow(xy_from, xy_to, color, style, lw):
    arrow = FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=8, color=color,
        lw=lw, linestyle=style, shrinkA=3.5, shrinkB=3.5, zorder=3,
    )
    fig.add_artist(arrow)


GIVEN_KW = dict(color=GIVEN_COLOR, style="solid", lw=1.0)


def found_kw(col):
    return dict(color=col, style=(0, (3.0, 1.4)), lw=1.4)


for i, mode in enumerate(MODES):
    row_top = ROW_TOP - i * ROW_H
    row_mid = row_top - 0.40
    col = mode["color"]
    found_role = mode["found_role"]

    fig.text(fx(X_MODE), fy(row_mid + 0.135), mode["name"], fontsize=FS_MODE, color=col,
              ha="left", va="center", fontweight="bold")
    fig.text(fx(X_MODE), fy(row_mid - 0.090), mode["tag"], fontsize=FS_TAG, color=TITLE_COLOR,
              ha="left", va="center", style="italic", linespacing=1.4)
    fig.text(fx(X_GIVEN), fy(row_mid), mode["given"], fontsize=FS_GIVEN, color=SIGNAL_COLOR,
              ha="left", va="center")
    fig.text(fx(X_FINDS), fy(row_mid), mode["finds"], fontsize=FS_FINDS, color=col,
              ha="left", va="center", fontweight="bold")

    for j, (role, text) in enumerate(mode["lines"]):
        y = row_top - 0.13 - j * 0.145
        is_found = role == found_role
        tag = f"∴ {role}" if is_found else role
        tag_color = col if is_found else GIVEN_COLOR
        fig.text(fx(X_SYL_TAG), fy(y), tag, fontsize=FS_SYL, color=tag_color, ha="left", va="top",
                  fontweight="bold" if is_found else "normal")
        fig.text(fx(X_SYL_TXT), fy(y), text, fontsize=FS_SYL, color=SIGNAL_COLOR,
                  ha="left", va="top")

    # --- the triangle, straight arrows, vertically centred on the row ---
    base_y = row_mid - APEX_H / 2
    apex_y = base_y + APEX_H
    A = (fx(X_DIAG0 + 0.15 * DIAG_W), fy(base_y))
    C = (fx(X_DIAG0 + 0.85 * DIAG_W), fy(base_y))
    B = (fx(X_DIAG0 + 0.50 * DIAG_W), fy(apex_y))

    name = mode["name"]
    f_kw = found_kw(col) if name == "Abduction" else GIVEN_KW
    g_kw = found_kw(col) if name == "Induction" else GIVEN_KW
    h_kw = found_kw(col) if name == "Deduction" else GIVEN_KW

    draw_arrow(A, B, **f_kw)
    draw_arrow(B, C, **g_kw)
    draw_arrow(A, C, **h_kw)

    for pt in (A, B, C):
        ax.plot(pt[0], pt[1], "o", color=SIGNAL_COLOR, markersize=3.5, zorder=4)
    fig.text(A[0] - fx(0.07), A[1], "A", fontsize=FS_VERTEX, color=SIGNAL_COLOR,
              ha="right", va="center")
    fig.text(A[0], A[1] - fy(0.075), "these beans", fontsize=FS_VLABEL, color=TITLE_COLOR,
              ha="center", va="top", style="italic")
    fig.text(B[0], B[1] + fy(0.05), "B", fontsize=FS_VERTEX, color=SIGNAL_COLOR,
              ha="center", va="bottom")
    fig.text(B[0], B[1] + fy(0.155), "bag beans", fontsize=FS_VLABEL, color=TITLE_COLOR,
              ha="center", va="bottom", style="italic")
    fig.text(C[0] + fx(0.07), C[1], "C", fontsize=FS_VERTEX, color=SIGNAL_COLOR,
              ha="left", va="center")
    fig.text(C[0], C[1] - fy(0.075), "white", fontsize=FS_VLABEL, color=TITLE_COLOR,
              ha="center", va="top", style="italic")

    f_label = "f" + (f"  ({mode['f_note']})" if mode.get("f_note") else "")
    g_label = "g" + (f"  ({mode['g_note']})" if mode.get("g_note") else "")
    fig.text((A[0] + B[0]) / 2 - fx(0.05), (A[1] + B[1]) / 2, f_label,
              fontsize=FS_EDGE, color=f_kw["color"], ha="right", va="center")
    fig.text((B[0] + C[0]) / 2 + fx(0.05), (B[1] + C[1]) / 2, g_label,
              fontsize=FS_EDGE, color=g_kw["color"], ha="left", va="center")
    fig.text((A[0] + C[0]) / 2, fy(base_y - 0.045), "h", fontsize=FS_EDGE,
              color=h_kw["color"], ha="center", va="top")

    if i < len(MODES) - 1:
        sep_y = row_top - ROW_H + 0.09
        fig.add_artist(plt.Line2D([fx(RULE_X0), fx(RULE_X1)], [fy(sep_y)] * 2,
                                  color="#e5e5e5", lw=0.6))

# The full-canvas axes makes bbox_inches="tight" a no-op, so the space vacated
# by the stripped title and footer would survive into the PDF. Crop to the union
# of the artists that actually draw something instead.
def artwork_bbox(pad=0.04):
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in fig.texts]
    boxes += [a.get_window_extent(renderer) for a in fig.artists]
    boxes += [a.get_window_extent(renderer) for a in ax.lines]
    bb = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


fig.canvas.draw()
BBOX = artwork_bbox()
save_figure(fig, 5, preview=preview_requested(), bbox_inches=BBOX)
