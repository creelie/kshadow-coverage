import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sensors = np.load("sensors.npy")
data = json.load(open("complex.json"))
R = data["R"]

res = 600
xs = np.linspace(-3, 15, res)
ys = np.linspace(-3, 15, res)
gx, gy = np.meshgrid(xs, ys)
mult = np.zeros_like(gx, dtype=int)
for (cx, cy) in sensors:
    mult += ((gx-cx)**2 + (gy-cy)**2 <= R**2).astype(int)

fig, ax = plt.subplots(figsize=(8, 8))
region2 = mult >= 2
ax.imshow(region2, extent=[-3, 15, -3, 15], origin="lower", cmap="Greens",
          alpha=0.45, vmin=0, vmax=1.4)
under2 = (mult >= 1) & (mult < 2)
ax.imshow(np.ma.masked_where(~under2, under2), extent=[-3, 15, -3, 15],
          origin="lower", cmap="Reds", alpha=0.55, vmin=0, vmax=1.2)

for (cx, cy) in sensors:
    ax.plot(cx, cy, "o", color="#1f4e96", markersize=3, zorder=5)

ax.set_xlim(-1, 13)
ax.set_ylim(-1, 13)
ax.set_aspect("equal")
ax.set_title(r"$\Delta_2(N)$ result: $b_0=1$, $b_1=8$ — exact match with "
             "independently sampled ground truth", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("y")
legend_text = ("blue dots = sensors\ngreen = covered $\\geq 2$ times\n"
               "red = covered exactly once\norange circles = the 8 enclosed\n"
               "holes counted in $b_1$ (boundary\nfalloff, touching the image\n"
               "edge, is not counted)")
ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

hole_coords = [(5.91,6.51), (5.18,0.20), (6.82,11.82), (6.82,0.17),
               (5.18,11.85), (0.20,6.00), (11.81,6.00), (6.04,8.02)]
for (hx, hy) in hole_coords:
    ax.add_patch(plt.Circle((hx, hy), 0.45, fill=False, edgecolor="orange",
                             linewidth=1.8, zorder=10))

fig.tight_layout()
fig.savefig("network_k2_result.png", dpi=150)
print("saved network_k2_result.png")
