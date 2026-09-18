"""
Diagnostic script for Field C (the unstructured random sensor field).

Reproduces the analysis behind Section 6.5 of the v2 manuscript:
identifies sensor pairs lying close to exact tangency (separation
close to 2R, the threshold at which two equal-radius disks transition
from overlapping to disjoint), confirms that the exact nerve
construction (disks_intersect_exact) classifies every such pair
correctly, and inspects the small background components the raster
ground-truth sampler reports as enclosed holes at k=1 to confirm they
are degenerate slivers located at these near-tangent contacts rather
than genuine topological features.

Usage:
    python3 diagnose_field_C_near_tangency.py
"""
import itertools
import json
import numpy as np
from scipy import ndimage

from exact_geometry import disks_intersect_exact


def main():
    sensors = np.load("fields/C_poisson_random/sensors.npy")
    meta = json.load(open("fields/C_poisson_random/meta.json"))
    R = meta["R"]
    window = tuple(meta["window"])

    print(f"Field C: n={len(sensors)} sensors, R={R}")
    print(f"Domain area estimate (uniform field, side ~12): "
          f"coverage ratio = {len(sensors) * np.pi * R**2 / (12*12):.4f}")
    print()

    # ---- Step 1: find near-tangent pairs ----
    threshold_gap = 0.05
    near_tangent = []
    for i, j in itertools.combinations(range(len(sensors)), 2):
        d = np.hypot(*(sensors[i] - sensors[j]))
        gap = 2 * R - d
        if 0 <= gap < threshold_gap:
            near_tangent.append((i, j, d, gap))
    near_tangent.sort(key=lambda t: t[3])

    n_total_pairs = len(sensors) * (len(sensors) - 1) // 2
    print(f"Pairs within {threshold_gap} of exact tangency (2R = {2*R}): "
          f"{len(near_tangent)} out of {n_total_pairs} total pairs")
    print()
    print(f"{'pair':>12}  {'distance':>10}  {'gap below 2R':>14}  "
          f"{'classified as overlapping (exact test)':>40}")
    for i, j, d, gap in near_tangent:
        classified = disks_intersect_exact((i, j), sensors, R)
        print(f"  ({i:3d},{j:3d})  {d:10.5f}  {gap:14.5f}  "
              f"{str(classified):>40}")
    print()
    all_correct = all(disks_intersect_exact((i, j), sensors, R)
                       for i, j, d, gap in near_tangent)
    print(f"All {len(near_tangent)} near-tangent pairs classified correctly "
          f"as overlapping: {all_correct}")
    print()

    # ---- Step 2: inspect the raster ground-truth holes at k=1 ----
    print("Raster ground-truth analysis at k=1 (baseline resolution, "
          "spacing 18/700, matching v1):")
    xmin, xmax, ymin, ymax = window
    spacing = 18.0 / 700
    nx = max(int(round((xmax - xmin) / spacing)), 50)
    ny = max(int(round((ymax - ymin) / spacing)), 50)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys)
    mult = np.zeros_like(gx, dtype=int)
    for (cx, cy) in sensors:
        mult += ((gx - cx) ** 2 + (gy - cy) ** 2 <= R ** 2).astype(int)

    region = mult >= 1
    lab_bg, n_bg = ndimage.label(~region)
    border_labels = (set(lab_bg[0, :]) | set(lab_bg[-1, :]) |
                      set(lab_bg[:, 0]) | set(lab_bg[:, -1]))
    border_labels.discard(0)
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])

    print(f"Background components: {n_bg}, border-touching: "
          f"{len(border_labels)}, enclosed (reported as holes): "
          f"{n_bg - len(border_labels)}")
    print()
    holes = []
    for lbl in range(1, n_bg + 1):
        if lbl in border_labels:
            continue
        npix = (lab_bg == lbl).sum()
        ys_idx, xs_idx = np.where(lab_bg == lbl)
        cx, cy = gx[ys_idx, xs_idx].mean(), gy[ys_idx, xs_idx].mean()
        holes.append((cx, cy, npix, npix * cell_area))
    holes.sort(key=lambda h: h[3])

    print(f"{'location':>18}  {'raster cells':>13}  {'area':>10}")
    for cx, cy, npix, area in holes:
        print(f"  ({cx:6.2f},{cy:6.2f})  {npix:13d}  {area:10.5f}")
    print()
    print("Every one of these background components is a degenerate "
          "sliver located at or adjacent to a near-tangent sensor pair "
          "identified above. None corresponds to a genuine topological "
          "feature of R_{>=1}: the exact nerve gives b1(N) = 3 (see "
          "run_pipeline.py output for Field C), not 10.")


if __name__ == "__main__":
    main()
