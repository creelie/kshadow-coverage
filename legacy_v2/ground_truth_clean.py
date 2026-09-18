"""
Ground truth, done the clean way: for region_k = {x : multiplicity(x) >= k},
compute its actual topology directly --
  b0_ground = number of connected components of region_k
  b1_ground = number of connected components of the COMPLEMENT of region_k
              that do NOT touch the border of the sampling window (i.e.
              genuine enclosed holes, excluding the unbounded exterior).
No fence/cutoff reasoning needed: region_k is automatically bounded, since
outside the union of all disks, multiplicity is 0 < k.
"""
import json
import numpy as np
from scipy import ndimage

sensors = np.load("sensors.npy")
data = json.load(open("complex.json"))
R = data["R"]

res = 700
xs = np.linspace(-3, 15, res)
ys = np.linspace(-3, 15, res)
gx, gy = np.meshgrid(xs, ys)

mult = np.zeros_like(gx, dtype=int)
for (cx, cy) in sensors:
    mult += ((gx-cx)**2 + (gy-cy)**2 <= R**2).astype(int)

def region_topology(region):
    lab_fg, n_fg = ndimage.label(region)
    lab_bg, n_bg = ndimage.label(~region)
    border_labels = set(lab_bg[0, :]) | set(lab_bg[-1, :]) | \
                    set(lab_bg[:, 0]) | set(lab_bg[:, -1])
    border_labels.discard(0)
    n_holes = sum(1 for lbl in range(1, n_bg+1) if lbl not in border_labels)
    hole_info = []
    for lbl in range(1, n_bg+1):
        if lbl in border_labels:
            continue
        ys_idx, xs_idx = np.where(lab_bg == lbl)
        cx, cy = gx[ys_idx, xs_idx].mean(), gy[ys_idx, xs_idx].mean()
        area = (lab_bg == lbl).sum() * (xs[1]-xs[0]) * (ys[1]-ys[0])
        hole_info.append((cx, cy, area))
    return n_fg, n_holes, hole_info

results = {}
for k in (1, 2, 3):
    region = mult >= k
    b0, b1, holes = region_topology(region)
    print(f"\nk={k}: region {{m>={k}}}  ->  b0 = {b0} component(s), "
          f"b1 = {b1} enclosed hole(s)")
    for (cx, cy, area) in sorted(holes, key=lambda h: -h[2]):
        print(f"    hole at ({cx:.2f},{cy:.2f}), area {area:.4f}")
    results[k] = {"b0": int(b0), "b1": int(b1)}

json.dump(results, open("ground_truth_clean.json", "w"))
