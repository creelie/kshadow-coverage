"""
Generalized multiplicity-field visualization, extending visualize_result.py
from the original v1/Nature Physics draft package to any of the five v2
fields. Reproduces the same visual language (green = covered >= 2, red =
covered exactly once, blue dots = sensors) so the new figures sit
naturally alongside the original network_k2_result.png if both are used
together.

Usage:
    python3 visualize_field.py A_ring_grid_seed3
    python3 visualize_field.py B_ring_grid_seed7
    python3 visualize_field.py D_hex_dead_zone
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELD_DIR_ROOT = "fields"


def load_field(name):
    sensors = np.load(f"{FIELD_DIR_ROOT}/{name}/sensors.npy")
    meta = json.load(open(f"{FIELD_DIR_ROOT}/{name}/meta.json"))
    result = json.load(open(f"{FIELD_DIR_ROOT}/{name}/result.json"))
    return sensors, meta, result


def find_holes(mult, k, xs, ys):
    from scipy import ndimage
    region = mult >= k
    lab_bg, n_bg = ndimage.label(~region)
    border_labels = (set(lab_bg[0, :]) | set(lab_bg[-1, :]) |
                      set(lab_bg[:, 0]) | set(lab_bg[:, -1]))
    border_labels.discard(0)
    gx, gy = np.meshgrid(xs, ys)
    holes = []
    for lbl in range(1, n_bg + 1):
        if lbl in border_labels:
            continue
        ys_idx, xs_idx = np.where(lab_bg == lbl)
        cx, cy = gx[ys_idx, xs_idx].mean(), gy[ys_idx, xs_idx].mean()
        npix = (lab_bg == lbl).sum()
        holes.append((cx, cy, npix))
    return holes


def visualize(name, k=2, out_path=None, res=700, title=None):
    sensors, meta, result = load_field(name)
    R = meta["R"]
    window = meta["window"]
    xmin, xmax, ymin, ymax = window

    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    gx, gy = np.meshgrid(xs, ys)
    mult = np.zeros_like(gx, dtype=int)
    for (cx, cy) in sensors:
        mult += ((gx - cx) ** 2 + (gy - cy) ** 2 <= R ** 2).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    region_k = mult >= k
    ax.imshow(region_k, extent=[xmin, xmax, ymin, ymax], origin="lower",
              cmap="Greens", alpha=0.45, vmin=0, vmax=1.4)
    under_k = (mult >= 1) & (mult < k)
    ax.imshow(np.ma.masked_where(~under_k, under_k),
              extent=[xmin, xmax, ymin, ymax], origin="lower",
              cmap="Reds", alpha=0.55, vmin=0, vmax=1.2)

    for (cx, cy) in sensors:
        ax.plot(cx, cy, "o", color="#1f4e96", markersize=3, zorder=5)

    pad = 1.0
    sx_min, sx_max = sensors[:, 0].min() - pad, sensors[:, 0].max() + pad
    sy_min, sy_max = sensors[:, 1].min() - pad, sensors[:, 1].max() + pad
    ax.set_xlim(sx_min, sx_max)
    ax.set_ylim(sy_min, sy_max)
    ax.set_aspect("equal")

    holes = find_holes(mult, k, xs, ys)
    for (hx, hy, npix) in holes:
        ax.add_patch(plt.Circle((hx, hy), 0.35, fill=False,
                                 edgecolor="orange", linewidth=1.6,
                                 zorder=10))

    b1_delta = result.get("delta2", {}).get("b1") if k == 2 else \
        result.get("delta1", {}).get("b1")
    b0_delta = result.get("delta2", {}).get("b0") if k == 2 else \
        result.get("delta1", {}).get("b0")
    field_label = name.replace("_", " ")
    if title is None:
        title = (rf"Field {name[0]}: $\Delta_{k}(N)$ result "
                  rf"$b_0={b0_delta}$, $b_1={b1_delta}$  "
                  rf"({len(holes)} background components shown)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    legend_text = (f"blue dots = {len(sensors)} sensors\n"
                    f"green = covered $\\geq {k}$ times\n"
                    f"red = covered 1..{k-1} times\n"
                    f"orange circles = enclosed background\ncomponents")
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.tight_layout()
    if out_path is None:
        out_path = f"figures/{name}_k{k}_result.png"
    fig.savefig(out_path, dpi=170)
    print(f"saved {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "A_ring_grid_seed3"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    visualize(name, k=k)
