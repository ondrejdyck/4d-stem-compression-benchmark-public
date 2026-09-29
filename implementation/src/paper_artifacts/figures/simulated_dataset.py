"""The simulated 4D-STEM dataset figure of the Discussion.

Detector panels are plotted in mrad, not pixel indices: the array is 147x169
because the orthogonal WSe2 supercell is 17.0434 x 19.68 A (ratio 2/sqrt3),
which makes the reciprocal-space pixel anisotropic (2.450 x 2.122 mrad).
On pixel indices the circular aperture would render as an ellipse.
"""
import json
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LogNorm

# Embed TrueType rather than matplotlib's default Type 3 fonts. Type 3 is
# rejected by several journals' production systems and carries no ToUnicode
# map, so text in the figure cannot be selected, searched or read aloud.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# matplotlib.path.Path is imported below, so pathlib is referenced by module.
NPZ = pathlib.Path(
    os.environ.get("FIGURE_DATA_DIR", pathlib.Path.home() / "4dstem-figure-data")
) / "wse2_pristine_128x128_374e.npz"
d = np.load(NPZ, allow_pickle=True)
m = json.loads(str(d["metadata_json"]))

adf_c = d["adf_from_counts"]
cnt = d["pattern_counts"][0]                       # tungsten column
lam = d["pattern_lambda"][0].astype(float)         # the same position
theta = d["theta_rad"] * 1e3
inner, outer = m["adf_inner_rad"] * 1e3, m["adf_outer_rad"] * 1e3
box = d["box_A"]
pos = int(d["pattern_index"][0])
pos_xy = d["scan_positions"][pos]
apos, asp = d["atom_positions"], d["atom_species"]

theta_x, theta_y = d["theta_x_rad"] * 1e3, d["theta_y_rad"] * 1e3
tx = theta[theta.shape[0] // 2, 0]
ty = theta[0, theta.shape[1] // 2]
EXT = [-ty, ty, -tx, tx]

W_COL, SE_COL = "#2f3b47", "#d99a2b"
PROBE, REDUCE = "#c2185b", "#0e7490"   # sampling / reduction
REDUCE_FILL = "#5cc8e8"                # lifts off the black panels
FS_T, FS_L = 8.5, 7.2

fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.05))
plt.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.26)

# ---------------------------------------------------------------- (a) specimen
ax = axes[0]
for dx in (-box[0], 0.0, box[0]):
    for dy in (-box[1], 0.0, box[1]):
        for big, species, col in ((0.62, "W", W_COL), (0.40, "Se", SE_COL)):
            for i in range(len(apos)):
                if (asp[i] == "W") != (species == "W"):
                    continue
                ax.add_patch(plt.Circle((apos[i][0] + dx, apos[i][1] + dy), big,
                                        fc=col, ec="none", zorder=3 if big > 0.5 else 2))

# raster: full-width sweeps from the very top, with a dotted flyback arrow
# running diagonally from the end of one sweep back to the start of the next
SWEEPS = (0.42, 2.25, 4.08)
for k, y in enumerate(SWEEPS):
    ax.add_patch(FancyArrow(0.0, y, box[0], 0, width=0.05, head_width=0.55,
                            head_length=0.85, length_includes_head=True,
                            fc=PROBE, ec="none", zorder=7))
    if k + 1 < len(SWEEPS):
        ax.add_patch(FancyArrowPatch((box[0], y), (0.0, SWEEPS[k + 1]),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=PROBE, lw=0.7, ls=":", alpha=0.85,
                                     shrinkA=2, shrinkB=2, zorder=7))

ax.plot(pos_xy[0], pos_xy[1], marker="x", color=PROBE, mew=1.8, ms=7, zorder=8)

ax.set_xlim(0, box[0]); ax.set_ylim(0, box[1]); ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_facecolor("none")
ax.set_title("specimen and scan", fontsize=FS_T)

# ------------------------------------------------------- (b) expected intensity
# 83% of lambda sits inside the bright-field disk, and the scattered halo the
# ADF detector integrates is two to three decades below it. Shown on a log
# scale clipped at 0.05 so that halo is visible; on a linear scale the panel is
# a bright disk on black and the reason an ADF signal exists is invisible.
ax = axes[1]
ax.imshow(lam, cmap="inferno", norm=LogNorm(vmin=1e-3, vmax=0.05),
          extent=EXT, origin="lower", aspect="equal")
ax.set_title("expected intensity $\\lambda$", fontsize=FS_T)

# ------------------------------------------------------------- (c) one pattern
ax = axes[2]
# 347 electrons over 24,843 pixels is 1.4% occupancy: as a raster this panel
# is black with a few specks. The detections are drawn as points instead, area
# proportional to the count, so the draw reads as the discrete thing it is.
ax.imshow(np.zeros_like(cnt), cmap="gray", vmin=0, vmax=1,
          extent=EXT, origin="lower", aspect="equal")
_iy, _ix = np.nonzero(cnt)
ax.scatter(theta_y[_iy, _ix], theta_x[_iy, _ix], s=1.6 * cnt[_iy, _ix],
           c="#ffd166", linewidths=0, zorder=3)
# annulus in the reduction colour: the whole ring is what gets drawn down
for r in (inner, outer):
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color=REDUCE_FILL, lw=0.8,
                            ls="--", zorder=4))
ax.set_title("Poisson draw", fontsize=FS_T)

# ------------------------------------------------------------------ (d) ADF
ax = axes[3]
sp = d["scan_positions"]
sx0, sx1 = float(sp[:, 0].min()), float(sp[:, 0].max())
sy0, sy1 = float(sp[:, 1].min()), float(sp[:, 1].max())
ax.imshow(adf_c, cmap="gray", extent=[sx0, sx1, sy1, sy0], aspect="equal")
ax.plot(pos_xy[0], pos_xy[1], marker="x", color=REDUCE, mew=1.8, ms=7, zorder=5)
ax.plot([sx0 + 0.9, sx0 + 5.9], [sy1 - 1.1]*2, color="w", lw=2, solid_capstyle="butt")
ax.text(sx0 + 3.4, sy1 - 1.45, "5 Å", color="w", fontsize=FS_L, ha="center", va="bottom")
ax.set_title("ADF image", fontsize=FS_T)

for i, ax in enumerate(axes):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.0, 1.13, f"({'abcd'[i]})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")

# ---------------------------------------------------------------- connectors
def F(ax, x, y):
    return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))

def flare(pt, top, bot, color="#64748b", alpha=0.13, zorder=0):
    """Filled band from a point out to an edge, easing in and out horizontally."""
    (px, py), (tx, ty_), (bx, by) = pt, top, bot
    c = 0.45 * (tx - px)
    verts = [(px, py),
             (px + c, py), (tx - c, ty_), (tx, ty_),      # upper edge
             (bx, by),                                     # across the far edge
             (bx - c, by), (px + c, py), (px, py)]         # lower edge, back
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4]
    fig.add_artist(PathPatch(Path(verts, codes), fc=color, ec="none",
                             alpha=alpha, zorder=zorder))

# aspect="equal" only settles the axes boxes at draw time, so resolve the
# connector endpoints after a draw or they miss the frame by a few percent.
fig.canvas.draw()

a_x = F(axes[0], pos_xy[0], pos_xy[1])
b_lt, b_lb = F(axes[1], EXT[0], EXT[3]), F(axes[1], EXT[0], EXT[2])
d_x = F(axes[3], pos_xy[0], pos_xy[1])
flare(a_x, b_lt, b_lb, color=PROBE, alpha=0.12)

# (b) and (c) are the same probe position on the same detector; what separates
# them is the draw, so they get a plain arrow rather than a taper.
b_r = F(axes[1], EXT[1], 0.0)
c_l = F(axes[2], EXT[0], 0.0)
fig.add_artist(FancyArrowPatch(b_r, c_l, arrowstyle="-|>", mutation_scale=8,
                               color="#64748b", lw=0.9, shrinkA=3, shrinkB=3,
                               zorder=6))

# The right-hand taper is gathered from the ADF annulus itself, not the frame:
# what survives is the ring, not the pattern. Its mouth follows the outer
# circle, so it reads as a selection rather than a panel-to-panel connector.
def gather(pt, cx, cy, radius, ax_src, a0=78.0, a1=-78.0, n=40,
           color=REDUCE_FILL, alpha=0.34, zorder=3):
    t = np.radians(np.linspace(a0, a1, n))
    arc = [F(ax_src, cx + radius*np.cos(u), cy + radius*np.sin(u)) for u in t]
    c = 0.45 * (pt[0] - arc[0][0])
    verts = [pt,
             (pt[0] - c, pt[1]), (arc[0][0] + c, arc[0][1]), arc[0]]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    verts += list(arc[1:]); codes += [Path.LINETO] * (len(arc) - 1)
    verts += [(arc[-1][0] + c, arc[-1][1]), (pt[0] - c, pt[1]), pt]
    codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    fig.add_artist(PathPatch(Path(verts, codes), fc=color, ec="none",
                             alpha=alpha, zorder=zorder))

gather(d_x, 0.0, 0.0, outer, axes[2])

out = str(pathlib.Path(__file__).with_name("simulated_dataset.png"))
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", out)
