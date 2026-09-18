"""
Reproduces Figure 8 of the v2 manuscript: a two-panel figure showing
Field E's full sensor layout with the largest maximal clique
highlighted on the left, and that clique shown in isolation (with
every disk boundary drawn) on the right.

This is the figure-generation counterpart to
diagnose_field_E_scaling.py, which produces the numerical evidence
(clique size, candidate subset count); this script produces the
visual companion.

Usage:
    python3 make_figure_E_dense_cluster.py
Output:
    E_dense_cluster_detail.png
"""
import itertools
import json
from math import comb

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exact_geometry import disks_intersect_exact

FIELD_DIR = "fields/E_two_cluster"


def main():
    sensors = np.load(f"{FIELD_DIR}/sensors.npy")
    meta = json.load(open(f"{FIELD_DIR}/meta.json"))
    R = meta["R"]
    N = len(sensors)

    pairs = [(i, j) for i, j in itertools.combinations(range(N), 2)
             if np.hypot(*(sensors[i] - sensors[j])) <= 2 * R]
    edges = [e for e in pairs if disks_intersect_exact(e, sensors, R)]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    maximal_cliques = list(nx.find_cliques(G))
    largest = max(maximal_cliques, key=len)
    print("largest clique size:", len(largest))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))

    # left: full field, colour sensors by whether they're in the
    # largest clique
    ax = axes[0]
    for (cx, cy) in sensors:
        circ = plt.Circle((cx, cy), R, fill=True, facecolor="#9fc8a8",
                           edgecolor="none", alpha=0.35, zorder=1)
        ax.add_patch(circ)
    in_clique = set(largest)
    for idx, (cx, cy) in enumerate(sensors):
        if idx in in_clique:
            ax.plot(cx, cy, "o", color="#d9534f", markersize=5, zorder=5)
        else:
            ax.plot(cx, cy, "o", color="#1f4e96", markersize=3, zorder=4)
    pad = 1.0
    ax.set_xlim(sensors[:, 0].min() - pad, sensors[:, 0].max() + pad)
    ax.set_ylim(sensors[:, 1].min() - pad, sensors[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Field E: dense cluster + sparse halo ($n=95$).\n"
                 f"Red = the {len(largest)}-sensor maximal clique "
                 f"(all mutually overlapping).", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # right: just the clique subgraph, drawn as overlapping disks to
    # show density
    ax2 = axes[1]
    clique_pts = sensors[list(in_clique)]
    for (cx, cy) in clique_pts:
        circ = plt.Circle((cx, cy), R, fill=False, edgecolor="#d9534f",
                           linewidth=0.8, alpha=0.55, zorder=2)
        ax2.add_patch(circ)
    ax2.plot(clique_pts[:, 0], clique_pts[:, 1], "o", color="#d9534f",
              markersize=5, zorder=5)
    pad2 = R + 0.3
    ax2.set_xlim(clique_pts[:, 0].min() - pad2, clique_pts[:, 0].max() + pad2)
    ax2.set_ylim(clique_pts[:, 1].min() - pad2, clique_pts[:, 1].max() + pad2)
    ax2.set_aspect("equal")
    ax2.set_title(f"The {len(largest)}-sensor maximal clique alone.\n"
                  rf"$\binom{{{len(largest)}}}{{8}} \approx "
                  f"{comb(len(largest), 8)/1e6:.1f}\\times 10^6$ "
                  "candidate 8-subsets\nfor face enumeration from this "
                  "clique alone.", fontsize=10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    fig.tight_layout()
    fig.savefig("figures/E_dense_cluster_detail.png", dpi=170)
    print("saved E_dense_cluster_detail.png")


if __name__ == "__main__":
    main()
