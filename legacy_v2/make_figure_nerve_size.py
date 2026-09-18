"""
Reproduces Figure 3 of the v2 manuscript: a bar chart comparing nerve
size (vertices, edges, triangles, log scale) across the four fields
whose nerve construction completed (A, B, C, D).

Usage:
    python3 make_figure_nerve_size.py
Output:
    nerve_size_comparison.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    fields = ["A_ring_grid_seed3", "B_ring_grid_seed7",
              "C_poisson_random", "D_hex_dead_zone"]
    labels = ["A", "B", "C", "D"]

    nerve_v, nerve_e, nerve_t = [], [], []
    for f in fields:
        r = json.load(open(f"fields/{f}/result.json"))
        nerve_v.append(r["nerve"]["vertices"])
        nerve_e.append(r["nerve"]["edges"])
        nerve_t.append(r["nerve"]["triangles"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(fields))
    width = 0.25
    ax.bar(x - width, nerve_v, width, label="vertices $|V(N)|$",
           color="#8DA0CB")
    ax.bar(x, nerve_e, width, label="edges $|E(N)|$", color="#FC8D62")
    ax.bar(x + width, nerve_t, width, label="triangles",
           color="#66C2A5")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Field {l}" for l in labels])
    ax.set_ylabel("count (log scale)")
    ax.set_title("Nerve size across the four completed fields")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    for i, (v, e, t) in enumerate(zip(nerve_v, nerve_e, nerve_t)):
        ax.text(i - width, v * 1.15, str(v), ha="center", fontsize=8)
        ax.text(i, e * 1.15, str(e), ha="center", fontsize=8)
        ax.text(i + width, t * 1.15, str(t), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/nerve_size_comparison.png", dpi=170)
    print("saved nerve_size_comparison.png")


if __name__ == "__main__":
    main()
