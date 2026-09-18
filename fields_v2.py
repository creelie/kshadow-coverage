"""
Generate multiple synthetic sensor fields for the v2 multi-topology
validation of the k-shadow complex construction.

Field A ("ring-grid", seed 3) is exactly the v1 field, reproduced here
unchanged for continuity with the published v1 record (same fence
construction, same interior grid, same seed, same four removed
sensors, same R = 1.55).

Fields B-E are new, independent topologies built to stress-test the
construction under different qualitative deployment geometries:
  B: ring-grid, different seed (seed 7) -- same topology class as v1,
     tests sensitivity to jitter only.
  C: pure random Poisson field in a square, no engineered fence or
     gap -- tests the construction on an unstructured layout with
     boundary effects left untouched.
  D: hexagonal lattice with random perturbation and a deliberate
     rectangular dead zone -- tests a regular, non-circular topology.
  E: two-cluster field (dense cluster + sparse halo) -- tests strong
     density heterogeneity within a single field.

Every field is saved with its own sensors.npy-equivalent array and
metadata so the rest of the pipeline (build_complex2.py, delta2.py,
ground_truth_clean.py) can be pointed at any one of them.
"""
import numpy as np
import json
import os

OUT_ROOT = "fields"
os.makedirs(OUT_ROOT, exist_ok=True)


def ring_sensors(n, cx, cy, radius_to_center):
    return [(cx + radius_to_center * np.cos(2 * np.pi * k / n),
             cy + radius_to_center * np.sin(2 * np.pi * k / n))
            for k in range(n)]


def field_A_ring_grid_seed3():
    """Exact reproduction of the v1 field (seed 3)."""
    rng = np.random.default_rng(3)
    R = 1.55
    fence_outer = ring_sensors(24, 6.0, 6.0, 7.4)
    fence_inner = ring_sensors(22, 6.0, 6.0, 5.9)
    fence = fence_outer + fence_inner
    coords = [6.0 + 1.4 * k for k in range(-3, 4)]
    grid_pts = []
    for x in coords:
        for y in coords:
            jx, jy = rng.uniform(-0.08, 0.08, 2)
            grid_pts.append((x + jx, y + jy))
    grid_arr = np.array(grid_pts)
    center = np.array([6.0, 6.0])
    dists = np.hypot(grid_arr[:, 0] - center[0], grid_arr[:, 1] - center[1])
    remove_idx = set(np.argsort(dists)[:4])
    grid_pts = [p for i, p in enumerate(grid_pts) if i not in remove_idx]
    sensors = np.array(fence + grid_pts)
    # design window matches the original v1 ground_truth_clean.py exactly:
    # xs = linspace(-3, 15, 700), i.e. window = (-3, 15, -3, 15)
    window = (-3.0, 15.0, -3.0, 15.0)
    return sensors, R, "ring-grid (v1 reproduction, seed 3)", window


def field_B_ring_grid_seed7():
    """Same topology class as the v1 field, independent seed."""
    rng = np.random.default_rng(7)
    R = 1.55
    fence_outer = ring_sensors(24, 6.0, 6.0, 7.4)
    fence_inner = ring_sensors(22, 6.0, 6.0, 5.9)
    fence = fence_outer + fence_inner
    coords = [6.0 + 1.4 * k for k in range(-3, 4)]
    grid_pts = []
    for x in coords:
        for y in coords:
            jx, jy = rng.uniform(-0.08, 0.08, 2)
            grid_pts.append((x + jx, y + jy))
    grid_arr = np.array(grid_pts)
    center = np.array([6.0, 6.0])
    dists = np.hypot(grid_arr[:, 0] - center[0], grid_arr[:, 1] - center[1])
    remove_idx = set(np.argsort(dists)[:4])
    grid_pts = [p for i, p in enumerate(grid_pts) if i not in remove_idx]
    sensors = np.array(fence + grid_pts)
    window = (-3.0, 15.0, -3.0, 15.0)
    return sensors, R, "ring-grid (independent seed 7)", window


def field_C_poisson_random():
    """Unstructured Poisson-type random field, no engineered fence."""
    rng = np.random.default_rng(11)
    R = 1.4
    n = 95
    pts = rng.uniform(0.0, 12.0, size=(n, 2))
    window = (-3.0, 15.0, -3.0, 15.0)
    return pts, R, "uniform random field (n=95, seed 11)", window


def field_D_hexagonal_with_dead_zone():
    """Hexagonal lattice, jittered, with a deliberate rectangular gap."""
    rng = np.random.default_rng(13)
    R = 1.35
    spacing = 1.7
    pts = []
    rows = 9
    cols = 9
    for r in range(rows):
        for c in range(cols):
            x = c * spacing + (spacing / 2 if r % 2 else 0)
            y = r * spacing * np.sqrt(3) / 2
            jx, jy = rng.uniform(-0.06, 0.06, 2)
            pts.append((x + jx, y + jy))
    pts = np.array(pts)
    keep = ~((pts[:, 0] > 5.0) & (pts[:, 0] < 7.5) &
             (pts[:, 1] > 4.5) & (pts[:, 1] < 6.5))
    pts = pts[keep]
    xmin, xmax = pts[:, 0].min() - 3.0, pts[:, 0].max() + 3.0
    ymin, ymax = pts[:, 1].min() - 3.0, pts[:, 1].max() + 3.0
    window = (xmin, xmax, ymin, ymax)
    return pts, R, "hexagonal lattice with rectangular dead zone (seed 13)", window


def field_E_two_cluster():
    """Dense cluster plus a sparse halo: strong density heterogeneity."""
    rng = np.random.default_rng(17)
    R = 1.3
    dense = rng.normal(loc=[5.0, 5.0], scale=0.9, size=(55, 2))
    halo = rng.uniform(0.0, 11.0, size=(40, 2))
    pts = np.vstack([dense, halo])
    xmin, xmax = pts[:, 0].min() - 3.0, pts[:, 0].max() + 3.0
    ymin, ymax = pts[:, 1].min() - 3.0, pts[:, 1].max() + 3.0
    window = (xmin, xmax, ymin, ymax)
    return pts, R, "dense cluster + sparse halo (seed 17)", window


FIELDS = {
    "A_ring_grid_seed3": field_A_ring_grid_seed3,
    "B_ring_grid_seed7": field_B_ring_grid_seed7,
    "C_poisson_random": field_C_poisson_random,
    "D_hex_dead_zone": field_D_hexagonal_with_dead_zone,
    "E_two_cluster": field_E_two_cluster,
}


if __name__ == "__main__":
    summary = {}
    for name, fn in FIELDS.items():
        sensors, R, desc, window = fn()
        field_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(field_dir, exist_ok=True)
        np.save(os.path.join(field_dir, "sensors.npy"), sensors)
        meta = {"n_sensors": int(len(sensors)), "R": float(R),
                "description": desc, "window": list(window)}
        json.dump(meta, open(os.path.join(field_dir, "meta.json"), "w"),
                   indent=2)
        summary[name] = meta
        print(f"{name}: n={meta['n_sensors']}, R={meta['R']}, "
              f"window={meta['window']}, {desc}")
    json.dump(summary, open(os.path.join(OUT_ROOT, "summary.json"), "w"),
               indent=2)
