"""
Reproduces Figure 6 of the v2 manuscript: a two-panel figure showing
Field C's full coverage region R_{>=1} on the left, and a zoomed view
of one near-tangent sensor contact (with disk boundaries drawn
explicitly) on the right.

This is the figure-generation counterpart to
diagnose_field_C_near_tangency.py, which produces the numerical
evidence (the table of near-tangent pairs and degenerate raster
slivers); this script produces the visual companion.

Usage:
    python3 make_figure_C_near_tangent.py
Output:
    C_near_tangent_detail.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIELD_DIR = "fields/C_poisson_random"


def main():
    sensors = np.load(f"{FIELD_DIR}/sensors.npy")
    meta = json.load(open(f"{FIELD_DIR}/meta.json"))
    R = meta["R"]
    xmin, xmax, ymin, ymax = meta["window"]

    res = 900
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    gx, gy = np.meshgrid(xs, ys)
    mult = np.zeros_like(gx, dtype=int)
    for (cx, cy) in sensors:
        mult += ((gx - cx) ** 2 + (gy - cy) ** 2 <= R ** 2).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    # left panel: full field overview, with a box marking the zoom region
    ax = axes[0]
    region1 = mult >= 1
    ax.imshow(region1, extent=[xmin, xmax, ymin, ymax], origin="lower",
              cmap="Greens", alpha=0.55, vmin=0, vmax=1.4)
    for (cx, cy) in sensors:
        ax.plot(cx, cy, "o", color="#1f4e96", markersize=3, zorder=5)
    pad = 1.0
    sx_min, sx_max = sensors[:, 0].min() - pad, sensors[:, 0].max() + pad
    sy_min, sy_max = sensors[:, 1].min() - pad, sensors[:, 1].max() + pad
    ax.set_xlim(sx_min, sx_max)
    ax.set_ylim(sy_min, sy_max)
    ax.set_aspect("equal")

    # this zoom box is chosen to contain one of the near-tangent pairs
    # identified by diagnose_field_C_near_tangency.py; adjust if you
    # regenerate the field with a different random seed
    zoom_box = (6.5, 8.0, 4.2, 5.0)
    rect = mpatches.Rectangle((zoom_box[0], zoom_box[2]),
                                zoom_box[1] - zoom_box[0],
                                zoom_box[3] - zoom_box[2], fill=False,
                                edgecolor="orange", linewidth=2, zorder=10)
    ax.add_patch(rect)
    ax.set_title(r"Field C: $R_{\geq 1}$ (uniform random, $n=95$). "
                 r"$\Delta_1(N)$ gives $b_1=3$.", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # right panel: zoomed view of the near-tangent sliver region
    ax2 = axes[1]
    zxs = np.linspace(zoom_box[0], zoom_box[1], 700)
    zys = np.linspace(zoom_box[2], zoom_box[3], 700)
    zgx, zgy = np.meshgrid(zxs, zys)
    zmult = np.zeros_like(zgx, dtype=int)
    for (cx, cy) in sensors:
        zmult += ((zgx - cx) ** 2 + (zgy - cy) ** 2 <= R ** 2).astype(int)
    zregion = zmult >= 1
    ax2.imshow(zregion, extent=zoom_box, origin="lower", cmap="Greens",
               alpha=0.55, vmin=0, vmax=1.4)
    for (cx, cy) in sensors:
        if (zoom_box[0] - R <= cx <= zoom_box[1] + R and
                zoom_box[2] - R <= cy <= zoom_box[3] + R):
            ax2.plot(cx, cy, "o", color="#1f4e96", markersize=5, zorder=5)
            circ = plt.Circle((cx, cy), R, fill=False,
                               edgecolor="#1f4e96", linewidth=0.7,
                               alpha=0.6, zorder=4)
            ax2.add_patch(circ)
    ax2.set_xlim(zoom_box[0], zoom_box[1])
    ax2.set_ylim(zoom_box[2], zoom_box[3])
    ax2.set_aspect("equal")
    ax2.set_title("Zoom: near-tangent sliver cluster\n"
                  "(raster sampler reports spurious holes here;\n"
                  "exact nerve test classifies every contact correctly)",
                  fontsize=9.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    fig.tight_layout()
    fig.savefig("figures/C_near_tangent_detail.png", dpi=170)
    print("saved C_near_tangent_detail.png")


if __name__ == "__main__":
    main()
