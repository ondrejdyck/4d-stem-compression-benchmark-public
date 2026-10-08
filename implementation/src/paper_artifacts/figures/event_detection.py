"""
Event-based detector illustration for the Discussion.

An event-based detector (e.g. Timepix4) throws away the *entire*
analog trace, at every pixel, all the time. What is stored is not a
reduced measurement at all -- it is a symbolic claim about what happened
(time of arrival, time over threshold, pixel location), produced by
comparing the transient charge cascade against an assumed pulse shape and
a threshold. The record is driven by the physical process but is not the
raw data itself.

Layout (two tiers):
  - top panel: the one-time, transient analog pulse at the hit pixel,
    with the assumed/model pulse shape shown faint and dashed underneath
    the noisy "actual" response, threshold line, ToA and ToT annotated.
  - bottom-left: the same trace, greyed out and hatched with the same
    "discarded" visual language used in the ADC figure, labeled as such.
  - bottom-right: a small boxed "ticket" -- not a plot -- holding only
    the four stored fields. Deliberately drawn as a different *kind* of
    object (no axes, no curve) to signal it is an interpretation, not a
    trace.

Run with: python -m paper_artifacts.figures.event_detection, from implementation/src.
Writes paper/generated/figures/figure_7.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d  # noqa: F401 -- registers the 3d projection

# Embed TrueType rather than matplotlib's default Type 3 fonts. Type 3 is
# rejected by several journals' production systems and carries no ToUnicode
# map, so text in the figure cannot be selected, searched or read aloud.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from paper_artifacts.outputs import save_figure, preview_requested


plt.rcParams["mathtext.fontset"] = "dejavusans"  # match mathtext to the default sans-serif font

# ---------------------------------------------------------------------------
# Style constants.
# ---------------------------------------------------------------------------
SIGNAL_COLOR = "#333333"
MODEL_COLOR = "#8899bb"
SAMPLE_COLOR = "C0"
DISCARD_FACE = "#bbbbbb"
DISCARD_EDGE = "#888888"
THRESH_COLOR = "#a33"


def style_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# The transient: a bi-exponential charge-cascade pulse (fast rise, slower
# fall) riding on a noisy baseline. pulse_clean is the assumed/model shape;
# the noisy trace is "what actually happens" at the hit pixel.
# ---------------------------------------------------------------------------
T0, T1 = 0.0, 200.0  # ns
T0_HIT = 80.0
TAU_RISE, TAU_FALL = 3.0, 16.0
AMP = 1.0
THRESHOLD = 0.18
NOISE_SIGMA = 0.035
SEED = 7


def pulse_clean(t):
    t = np.asarray(t, dtype=float)
    x = np.clip(t - T0_HIT, 0, None)
    raw = np.exp(-x / TAU_FALL) - np.exp(-x / TAU_RISE)
    t_peak = T0_HIT + (TAU_RISE * TAU_FALL / (TAU_FALL - TAU_RISE)) * np.log(TAU_FALL / TAU_RISE)
    peak_val = np.exp(-(t_peak - T0_HIT) / TAU_FALL) - np.exp(-(t_peak - T0_HIT) / TAU_RISE)
    return AMP * raw / peak_val


t_dense = np.linspace(T0, T1, 4000)
clean = pulse_clean(t_dense)

_rng = np.random.default_rng(SEED)
noisy = clean + _rng.normal(0, NOISE_SIGMA, size=t_dense.size)

# --- Threshold crossings, computed from the clean/model shape (the model
# is what the detector actually compares against; noise jitters the exact
# crossing by less than a sample and would just clutter the illustration).
peak_idx = np.argmax(clean)
t_toa = np.interp(THRESHOLD, clean[:peak_idx + 1], t_dense[:peak_idx + 1])
seg_v = clean[peak_idx:][::-1]
seg_t = t_dense[peak_idx:][::-1]
t_tot_end = np.interp(THRESHOLD, seg_v, seg_t)
tot = t_tot_end - t_toa

HIT_X, HIT_Y = 37, 52

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 6.3))
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1.35, 1.0], hspace=0.55, top=0.87)
top_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[0], width_ratios=[1.0, 1.3], wspace=0.3,
)
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[1], width_ratios=[1.4, 1.0], wspace=0.28,
)

ax_array = fig.add_subplot(top_gs[0], projection="3d")
ax_pulse = fig.add_subplot(top_gs[1])
ax_discard = fig.add_subplot(bottom_gs[0])
ax_record = fig.add_subplot(bottom_gs[1])

# Title and footer note are omitted: in the manuscript that material belongs
# to the LaTeX caption, not to the artwork.

# ---------------------------------------------------------------------------
# Top-left panel: a real 3D plot. A plain grid (pixel columns x pixel rows)
# faces the viewer like a wall, sitting at depth=0. From the center of each
# cell, that pixel's data stream extends straight back in depth: noise for
# every pixel except the hit one, which gets the rise-and-fall pulse.
#
# mplot3d has no true per-pixel z-buffer, so instead of relying on
# depth-based occlusion (which only reveals traces that happen to drift
# past the wall's own outer edge), draw order is forced explicitly via
# computed_zorder=False: quiet streams are drawn first and veiled under a
# translucent wall, and the hit pixel's stream is drawn last, on top, so it
# reads clearly rather than being hidden by interior position.
# ---------------------------------------------------------------------------
N_COLS, N_ROWS = 6, 6
DEPTH_MAX = 1.0
NOISE_AMP_MINI = 0.06
TRACE_AMP_MINI = 0.42
HL_COL, HL_ROW = 3, 2

_rng_array = np.random.default_rng(23)
ax_array.computed_zorder = False


def mini_pulse(d, d0=0.34, tau_rise=0.02, tau_fall=0.11, amp=1.0):
    d = np.asarray(d, dtype=float)
    x = np.clip(d - d0, 0, None)
    raw = np.exp(-x / tau_fall) - np.exp(-x / tau_rise)
    d_peak = d0 + (tau_rise * tau_fall / (tau_fall - tau_rise)) * np.log(tau_fall / tau_rise)
    peak_val = np.exp(-(d_peak - d0) / tau_fall) - np.exp(-(d_peak - d0) / tau_rise)
    return amp * raw / peak_val


depth_local = np.linspace(0, DEPTH_MAX, 60)

# Quiet pixel streams, drawn first so the wall veils them.
for j in range(N_ROWS):
    for i in range(N_COLS):
        if i == HL_COL and j == HL_ROW:
            continue
        wig = _rng_array.normal(0, NOISE_AMP_MINI, size=depth_local.size)
        xs = np.full_like(depth_local, float(i))
        zs = j + wig
        ax_array.plot(xs, depth_local, zs, color="#999999", lw=0.8, zorder=1)

# Translucent front wall (the pixel grid), sitting at depth (y) = 0.
_grid_xx, _grid_zz = np.meshgrid([-0.5, N_COLS - 0.5], [-0.5, N_ROWS - 0.5])
ax_array.plot_surface(
    _grid_xx, np.zeros_like(_grid_xx), _grid_zz,
    color="#e5e5e5", alpha=0.55, shade=False, edgecolor="none", zorder=2,
)
for i in range(N_COLS + 1):
    xv = i - 0.5
    ax_array.plot([xv, xv], [0, 0], [-0.5, N_ROWS - 0.5], color="#aaaaaa", lw=0.7, zorder=3)
for j in range(N_ROWS + 1):
    zv = j - 0.5
    ax_array.plot([-0.5, N_COLS - 0.5], [0, 0], [zv, zv], color="#aaaaaa", lw=0.7, zorder=3)

# Highlight the hit pixel's own cell border in blue.
_hb_x0, _hb_x1 = HL_COL - 0.5, HL_COL + 0.5
_hb_z0, _hb_z1 = HL_ROW - 0.5, HL_ROW + 0.5
ax_array.plot(
    [_hb_x0, _hb_x1, _hb_x1, _hb_x0, _hb_x0], [0, 0, 0, 0, 0],
    [_hb_z0, _hb_z0, _hb_z1, _hb_z1, _hb_z0],
    color=SAMPLE_COLOR, lw=1.6, zorder=3.5,
)

# The hit pixel's stream, drawn in the same pass as the quiet ones (behind
# the wall, veiled the same way) but bold enough to still read clearly.
_hit_wig = _rng_array.normal(0, NOISE_AMP_MINI, size=depth_local.size) * 0.6
_hit_wig += mini_pulse(depth_local, amp=TRACE_AMP_MINI)
_hit_xs = np.full_like(depth_local, float(HL_COL))
_hit_zs = HL_ROW + _hit_wig
ax_array.plot(_hit_xs, depth_local, _hit_zs, color=SAMPLE_COLOR, lw=2.2, zorder=1)

ax_array.set_xlim(-0.5, N_COLS - 0.5)
ax_array.set_ylim(0, DEPTH_MAX)
ax_array.set_zlim(-0.5, N_ROWS - 0.5)
ax_array.set_box_aspect((N_COLS, N_ROWS * 0.7, N_ROWS))
ax_array.view_init(elev=22, azim=-40)
ax_array.set_axis_off()

title_array = ax_array.text2D(
    0.5, 1.05, "Every pixel is read out, continuously",
    transform=ax_array.transAxes, fontsize=12, color="#555555", ha="center",
)

# ---------------------------------------------------------------------------
# Top panel: the transient itself
# ---------------------------------------------------------------------------
pulse_ylim = (min(noisy.min(), -0.08), max(noisy.max(), AMP) + 0.15)

ax_pulse.axhline(THRESHOLD, color=THRESH_COLOR, lw=1.2, linestyle="--", zorder=2)
ax_pulse.text(
    T1 - 15, THRESHOLD, "threshold", color=THRESH_COLOR, fontsize=10.5,
    ha="right", va="bottom",
)

ax_pulse.plot(t_dense, clean, color=MODEL_COLOR, lw=1.3, linestyle="--", zorder=2)
ax_pulse.plot(t_dense, noisy, color=SIGNAL_COLOR, lw=1.1, zorder=3)

ax_pulse.annotate(
    "assumed pulse shape",
    xy=(T0_HIT + 45, pulse_clean(np.array([T0_HIT + 45]))[0]), xycoords="data",
    xytext=(T0_HIT + 55, 0.75), textcoords="data",
    fontsize=10.5, color=MODEL_COLOR, ha="left",
    arrowprops=dict(arrowstyle="-", color=MODEL_COLOR, lw=0.9),
)

# ToA marker
# The stem runs from the dot down past the axis line: the stretch that reads as
# a marker is then clear of the noisy baseline it would otherwise sit inside.
_toa_stem_bottom = pulse_ylim[0] - 0.16
ax_pulse.plot([t_toa, t_toa], [_toa_stem_bottom, THRESHOLD], color=SAMPLE_COLOR,
              lw=1.8, zorder=4, clip_on=False)
ax_pulse.scatter([t_toa], [THRESHOLD], color=SAMPLE_COLOR, s=45, zorder=5,
                  edgecolor="white", linewidth=0.8, clip_on=False)
ax_pulse.annotate(
    "time of arrival (ToA)", xy=(t_toa, _toa_stem_bottom), xycoords="data",
    xytext=(t_toa - 5, _toa_stem_bottom - 0.02), textcoords="data",
    fontsize=10.5, color=SAMPLE_COLOR, ha="right", va="top", clip_on=False,
    # xy now sits below the axes; without this matplotlib drops the annotation.
    annotation_clip=False,
)

# ToT bracket
bracket_y = THRESHOLD + 0.12
ax_pulse.annotate(
    "", xy=(t_tot_end, bracket_y), xytext=(t_toa, bracket_y),
    xycoords="data", textcoords="data",
    arrowprops=dict(arrowstyle="<->", color=SAMPLE_COLOR, lw=2.0, mutation_scale=16),
)
_tot_mid = 0.5 * (t_toa + t_tot_end)
ax_pulse.annotate(
    "time over threshold\n(ToT)", xy=(_tot_mid, bracket_y), xycoords="data",
    xytext=(T0_HIT - 12, bracket_y + 0.34), textcoords="data",
    fontsize=10.5, color=SAMPLE_COLOR, ha="right", va="center",
    arrowprops=dict(arrowstyle="-", color=SAMPLE_COLOR, lw=0.9),
)

ax_pulse.set_xlim(T0, T1)
ax_pulse.set_ylim(pulse_ylim)
style_axes(ax_pulse)
ax_pulse.set_xlabel("Time", fontsize=13, color="#333333")
title_pulse = ax_pulse.text(
    0.5, 1.1, "A brief, one-time transient at a single pixel",
    transform=ax_pulse.transAxes, fontsize=12, color="#555555", ha="center",
)

# ---------------------------------------------------------------------------
# Bottom-left: everything that gets thrown away
# ---------------------------------------------------------------------------
ax_discard.plot(t_dense, noisy, color=DISCARD_EDGE, lw=1.0, zorder=2)
rect_discard = Rectangle(
    (T0, pulse_ylim[0]), T1 - T0, pulse_ylim[1] - pulse_ylim[0],
    facecolor=DISCARD_FACE, alpha=0.45, edgecolor=DISCARD_EDGE,
    linewidth=1.0, hatch="////", zorder=1,
)
ax_discard.add_patch(rect_discard)
ax_discard.set_xlim(T0, T1)
ax_discard.set_ylim(pulse_ylim)
style_axes(ax_discard)
ax_discard.set_title("All analog traces discarded", fontsize=11.5, color="#555555")

# ---------------------------------------------------------------------------
# Bottom-right: the ticket -- not a plot, a small record
# ---------------------------------------------------------------------------
ax_record.set_xlim(0, 1)
ax_record.set_ylim(0, 1)
ax_record.axis("off")

card = FancyBboxPatch(
    (0.08, 0.12), 0.84, 0.62,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    facecolor="white", edgecolor=SIGNAL_COLOR, linewidth=1.3, zorder=2,
)
ax_record.add_patch(card)

record_lines = [
    f"ToA   =  {t_toa:6.1f} ns",
    f"ToT   =  {tot:6.1f} ns",
    f"x, y  =  ({HIT_X}, {HIT_Y})",
]
for i, line in enumerate(record_lines):
    ax_record.text(
        0.18, 0.60 - i * 0.15, line, family="monospace", fontsize=12.5,
        color=SIGNAL_COLOR, ha="left", va="center", zorder=3,
    )

ax_record.set_title(
    "Interpretation recorded", fontsize=11.5, color="#555555",
)

# ---------------------------------------------------------------------------
# Vertical compression pass. Row 0 (array, pulse) and row 1 (discard,
# record) are each flattened in place -- anchored at their own top edge, so
# titles/annotations built above stay put -- and row 1 is pulled up to
# close the gap row 0's shrink leaves behind, with that same inter-row gap
# also tightened. Doing this via set_position (rather than GridSpec
# height_ratios/hspace) is what actually shrinks the saved image: GridSpec
# only redistributes space within its fixed allotted region, while explicit
# positioning frees real margin for bbox_inches="tight" to crop away.
# ---------------------------------------------------------------------------
_pos_array = ax_array.get_position()
_pos_pulse = ax_pulse.get_position()
_pos_discard = ax_discard.get_position()
_pos_record = ax_record.get_position()

SCALE_ROW0 = 0.70
SCALE_ROW1 = 0.80
GAP_SCALE = 1.05  # keeps row 1 clear of the ToA label hanging below panel (b)

row0_top = _pos_array.y1
for _ax, _p in ((ax_array, _pos_array), (ax_pulse, _pos_pulse)):
    _new_h = _p.height * SCALE_ROW0
    _ax.set_position([_p.x0, row0_top - _new_h, _p.width, _new_h])
row0_new_bottom = row0_top - _pos_array.height * SCALE_ROW0

_orig_gap = _pos_array.y0 - _pos_discard.y1
row1_new_top = row0_new_bottom - _orig_gap * GAP_SCALE
for _ax, _p in ((ax_discard, _pos_discard), (ax_record, _pos_record)):
    _new_h = _p.height * SCALE_ROW1
    _ax.set_position([_p.x0, row1_new_top - _new_h, _p.width, _new_h])
row1_new_bottom = row1_new_top - _pos_discard.height * SCALE_ROW1

# ---------------------------------------------------------------------------
# Panel labels. Positioned from each axes' final (post-compression) box and
# hung just outside its top-left corner, clear of the centred panel titles.
# ---------------------------------------------------------------------------
# Each panel title is centred and wider than its own axes box, so a label hung
# at the box corner lands inside the title. Anchor above the title instead,
# left-aligned to whichever of the two reaches further left.
PANEL_LABEL_FS = 13
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_inv = fig.transFigure.inverted()
for _letter, _ax, _title in (
    ("a", ax_array, title_array),
    ("b", ax_pulse, title_pulse),
    ("c", ax_discard, ax_discard.title),
    ("d", ax_record, ax_record.title),
):
    _p = _ax.get_position()
    _tb = _title.get_window_extent(_rend).transformed(_inv)
    fig.text(
        min(_p.x0, _tb.x0) - 0.010, _tb.y1 + 0.010, f"({_letter})",
        fontsize=PANEL_LABEL_FS, fontweight="bold", color=SIGNAL_COLOR,
        ha="left", va="bottom",
    )

save_figure(fig, 7, preview=preview_requested())
