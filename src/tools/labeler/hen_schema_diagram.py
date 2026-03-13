"""
Generate an academic-style diagram of the hen pose schema.
Run: python src/tools/labeler/hen_schema_diagram.py
Outputs: hen_pose_schema.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
import numpy as np

# ── Schema order (matches schema.py indices) ─────────────────────────────────
# Index: name          color
SCHEMA_ORDER = [
    (0,  "tail_base",   "#4895EF"),  # blue
    (1,  "tail_tip",    "#ADE8F4"),  # light cyan
    (2,  "left_hock",   "#FF9F1C"),  # amber
    (3,  "right_hock",  "#9B5DE5"),  # violet
    (4,  "left_foot",   "#FFBF69"),  # light amber
    (5,  "right_foot",  "#C77DFF"),  # light violet
    (6,  "neck_back",   "#E9C46A"),  # golden yellow
    (7,  "middle_back", "#52B788"),  # sage green
    (8,  "comb",        "#F4A261"),  # warm orange
    (9,  "beak",        "#E63946"),  # crimson red
]

COLORS = {name: color for _, name, color in SCHEMA_ORDER}

# ── Keypoint positions (x, y) — figure coords [0..1] ────────────────────────
# Side-view: head on left, tail on right, ground at bottom
KP = {
    "beak":         (0.07, 0.47),
    "comb":         (0.12, 0.28),
    "neck_back":    (0.27, 0.28),
    "middle_back":  (0.52, 0.22),
    "tail_base":    (0.77, 0.30),
    "tail_tip":     (0.95, 0.20),
    "left_hock":    (0.40, 0.67),
    "right_hock":   (0.50, 0.71),
    "left_foot":    (0.33, 0.90),
    "right_foot":   (0.44, 0.90),
}

SKELETON = [
    ("beak",        "comb"),
    ("beak",        "neck_back"),
    ("neck_back",   "middle_back"),
    ("middle_back", "tail_base"),
    ("tail_base",   "tail_tip"),
    ("tail_base",   "left_hock"),
    ("tail_base",   "right_hock"),
    ("middle_back", "left_hock"),
    ("middle_back", "right_hock"),
    ("left_hock",   "left_foot"),
    ("right_hock",  "right_foot"),
]

# ── Hen body silhouette (approximate ellipses) ───────────────────────────────
def draw_hen_silhouette(ax):
    body_color = "#E8E0D5"
    edge_color = "#B0A090"
    lw = 1.2

    # Main body
    body = Ellipse((0.52, 0.44), width=0.60, height=0.34,
                   facecolor=body_color, edgecolor=edge_color,
                   linewidth=lw, zorder=1)
    ax.add_patch(body)

    # Head
    head = Ellipse((0.13, 0.42), width=0.16, height=0.20,
                   facecolor=body_color, edgecolor=edge_color,
                   linewidth=lw, zorder=1)
    ax.add_patch(head)

    # Neck connector (smooth quadrilateral)
    neck_x = [0.17, 0.25, 0.28, 0.20]
    neck_y = [0.33, 0.30, 0.50, 0.52]
    ax.fill(neck_x, neck_y, color=body_color, zorder=1)
    ax.plot(neck_x + [neck_x[0]], neck_y + [neck_y[0]],
            color=edge_color, lw=lw, zorder=1)

    # Tail feather (triangle)
    tail_x = [0.77, 0.97, 0.88]
    tail_y = [0.30, 0.18, 0.38]
    ax.fill(tail_x, tail_y, color=body_color, zorder=1)
    ax.plot(tail_x + [tail_x[0]], tail_y + [tail_y[0]],
            color=edge_color, lw=lw, zorder=1)

    # Comb (small triangles on head)
    comb_x = [0.10, 0.12, 0.14, 0.16, 0.18]
    comb_y = [0.27, 0.21, 0.27, 0.22, 0.28]
    ax.fill(comb_x, comb_y, color="#E63946", alpha=0.6, zorder=2)

    # Eye
    eye = plt.Circle((0.11, 0.40), 0.013, color="#555555", zorder=3)
    ax.add_patch(eye)
    eye_hl = plt.Circle((0.115, 0.398), 0.004, color="white", zorder=4)
    ax.add_patch(eye_hl)

    # Beak (triangle)
    beak_x = [0.04, 0.09, 0.09]
    beak_y = [0.47, 0.44, 0.50]
    ax.fill(beak_x, beak_y, color="#D4B483", zorder=2)
    ax.plot(beak_x + [beak_x[0]], beak_y + [beak_y[0]],
            color=edge_color, lw=0.8, zorder=2)

    # Legs (simple lines)
    for lx, ly, fx, fy in [
        (0.40, 0.60, 0.33, 0.90),   # left leg
        (0.50, 0.60, 0.44, 0.90),   # right leg
    ]:
        ax.plot([lx, fx], [ly, fy], color="#B0A090", lw=3, zorder=1, solid_capstyle="round")
        # Toes
        for dx in [-0.035, 0, 0.025]:
            ax.plot([fx, fx + dx], [fy, fy + 0.04],
                    color="#B0A090", lw=2, zorder=1, solid_capstyle="round")


# ── Main figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)   # y increases downward (image coords)
ax.set_aspect("equal")
ax.axis("off")

# Ground line
ax.axhline(0.94, color="#CCCCCC", lw=1.2, linestyle="--", zorder=0)

# Silhouette (drawn before keypoints)
draw_hen_silhouette(ax)

# ── Skeleton lines ────────────────────────────────────────────────────────────
for (a_name, b_name) in SKELETON:
    x0, y0 = KP[a_name]
    x1, y1 = KP[b_name]
    # Use average of the two endpoint colours
    c0 = np.array([int(COLORS[a_name][i:i+2], 16) for i in (1, 3, 5)]) / 255
    c1 = np.array([int(COLORS[b_name][i:i+2], 16) for i in (1, 3, 5)]) / 255
    avg_c = (c0 + c1) / 2
    ax.plot([x0, x1], [y0, y1], color=avg_c, lw=2.2, zorder=3,
            solid_capstyle="round", alpha=0.85)

# ── Keypoints ─────────────────────────────────────────────────────────────────
POINT_R = 0.022
for name, (x, y) in KP.items():
    color = COLORS[name]
    circle = plt.Circle((x, y), POINT_R, color=color,
                         zorder=5, linewidth=1.5,
                         edgecolor="white")
    ax.add_patch(circle)

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_OFFSET = {
    "beak":         (-0.00, +0.07),
    "comb":         (-0.02, -0.05),
    "neck_back":    (+0.00, -0.06),
    "middle_back":  (+0.01, -0.06),
    "tail_base":    (+0.00, -0.06),
    "tail_tip":     (+0.01, -0.05),
    "left_hock":    (-0.09, +0.00),
    "right_hock":   (+0.04, +0.00),
    "left_foot":    (-0.09, +0.04),
    "right_foot":   (+0.04, +0.04),
}

for name, (x, y) in KP.items():
    dx, dy = LABEL_OFFSET.get(name, (0.03, 0.03))
    display = name.replace("_", "\u200b_")  # zero-width space keeps wrapping clean
    ax.text(x + dx, y + dy, name.replace("_", " "),
            fontsize=7.5, ha="center", va="center",
            color="#222222",
            fontfamily="DejaVu Sans",
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COLORS[name], linewidth=1.2, alpha=0.90),
            zorder=6)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=color, edgecolor="white", linewidth=0.5,
                   label=f"[{idx}] {name.replace('_', ' ')}")
    for idx, name, color in SCHEMA_ORDER
]
leg = ax.legend(
    handles=legend_handles,
    title="Keypoints",
    title_fontsize=8,
    fontsize=7.5,
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    handlelength=1.2,
    handleheight=1.2,
    ncol=2,
    columnspacing=1.0,
    bbox_to_anchor=(1.00, 0.02),
)
leg._legend_box.align = "left"

# ── Title & caption ───────────────────────────────────────────────────────────
ax.set_title("Laying Hen — Pose Estimation Schema (10 Keypoints)",
             fontsize=13, fontweight="bold", pad=12,
             color="#1A1A2E", fontfamily="DejaVu Sans")
ax.text(0.50, 0.995, "Side view  ·  Left limbs: amber  ·  Right limbs: violet  ·  Spine: red → green → blue",
        ha="center", va="bottom", fontsize=7, color="#888888",
        transform=ax.transAxes)

plt.tight_layout()
out = "hen_pose_schema.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
