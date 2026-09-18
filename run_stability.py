"""
run_stability.py -- the two stability statements of the paper, checked.

Both concern how far the certificate can move when the input moves, and both
are what a clinical reading of the barcode needs: contact positions come from
a co-registration that is good to a millimetre or two, and contacts fail.

  (i)  Contact loss.  Removing m contacts moves every bar of the redundancy
       barcode by at most m levels of k.  The script removes m contacts at
       random, recomputes the barcode of the subdivision filtration, and
       measures the bottleneck distance to the barcode of the full array.

  (ii) Localisation.  Moving every contact by at most delta interleaves the
       radius filtrations of the two placements by delta, so the Betti number
       of the perturbed placement at radius r lies between the two values of
       the unperturbed placement at r - delta and r + delta.  The script
       perturbs every centre by exactly delta in a random direction and
       compares the three curves.

The bottleneck distance is computed exactly, by binary search over the
finitely many candidate thresholds with a bipartite matching at each, the
diagonal counted with the usual half-length rule.

Writes results/stability.json.
"""
import json, os, sys, itertools
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields_v2 import FIELDS
from kshadow import nerve_of_disks, subdivision_persistence, betti_from_bars, shadow_betti

os.makedirs('results', exist_ok=True)


# ---------------------------------------------------------------- bottleneck
def _cost(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _diag(p):
    return abs(p[1] - p[0]) / 2.0


def bottleneck(A, B):
    """
    Exact bottleneck distance between two finite persistence diagrams, points
    given as (birth, death) pairs.  Candidate thresholds are the pairwise
    L-infinity costs and the half-lengths to the diagonal; for each candidate
    a perfect matching of the bipartite graph that allows any point to go to
    the diagonal decides feasibility, and the least feasible candidate is the
    answer.
    """
    A, B = list(A), list(B)
    if not A and not B:
        return 0.0
    cand = {0.0}
    for p in A:
        cand.add(_diag(p))
    for q in B:
        cand.add(_diag(q))
    for p in A:
        for q in B:
            cand.add(_cost(p, q))
    cand = sorted(cand)

    def feasible(eps):
        G = nx.Graph()
        left = [('a', i) for i in range(len(A))]
        right = [('b', j) for j in range(len(B))]
        # a diagonal partner for every point on the other side
        dl = [('db', j) for j in range(len(B))]
        dr = [('da', i) for i in range(len(A))]
        G.add_nodes_from(left + dl, bipartite=0)
        G.add_nodes_from(right + dr, bipartite=1)
        for i, p in enumerate(A):
            for j, q in enumerate(B):
                if _cost(p, q) <= eps:
                    G.add_edge(('a', i), ('b', j))
            if _diag(p) <= eps:
                G.add_edge(('a', i), ('da', i))
        for j, q in enumerate(B):
            if _diag(q) <= eps:
                G.add_edge(('db', j), ('b', j))
        for i in range(len(A)):
            for j in range(len(B)):
                G.add_edge(('db', j), ('da', i))
        m = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
        matched = set()
        for u, v in m:
            matched.add(u); matched.add(v)
        return all(x in matched for x in left + right)

    lo, hi = 0, len(cand) - 1
    if not feasible(cand[hi]):
        return float(cand[hi])
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(cand[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])


LOC_N = 30
LOC_DELTA = 0.08


def mec_radius(pts):
    """
    Radius of the smallest enclosing circle of a small point set, exactly.
    A smallest enclosing circle is determined either by two of the points,
    as a diameter, or by three of them, as a circumcircle, so enumerating
    both and taking the smallest circle that contains everything is exact
    for the sets of at most six points that occur here.  The library's
    own helper takes at most three points.
    """
    pts = np.asarray(pts, float)
    m = len(pts)
    if m == 1:
        return 0.0
    best = float('inf')
    for i in range(m):
        for j in range(i + 1, m):
            c = 0.5 * (pts[i] + pts[j])
            r = float(np.linalg.norm(pts[i] - c))
            if np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-12):
                best = min(best, r)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                a, b, c0 = pts[i], pts[j], pts[k]
                d = 2.0 * (a[0] * (b[1] - c0[1]) + b[0] * (c0[1] - a[1])
                           + c0[0] * (a[1] - b[1]))
                if abs(d) < 1e-14:
                    continue
                ux = ((a @ a) * (b[1] - c0[1]) + (b @ b) * (c0[1] - a[1])
                      + (c0 @ c0) * (a[1] - b[1])) / d
                uy = ((a @ a) * (c0[0] - b[0]) + (b @ b) * (a[0] - c0[0])
                      + (c0 @ c0) * (b[0] - a[0])) / d
                cen = np.array([ux, uy])
                r = float(np.linalg.norm(a - cen))
                if np.all(np.linalg.norm(pts - cen, axis=1) <= r + 1e-12):
                    best = min(best, r)
    return best


def radius_barcode(P, k, rmax):
    """
    Persistence of the radius filtration of the k-shadow complex of disks of
    a common radius about the points P, as the 2-skeleton with each face
    carrying the radius at which it appears.
    """
    from kshadow import persistence_2complex
    P = np.asarray(P, float)
    m = len(P)
    ent = {}

    def enter(idx):
        idx = tuple(sorted(idx))
        if idx not in ent:
            ent[idx] = mec_radius(P[list(idx)])
        return ent[idx]

    verts = [c for c in itertools.combinations(range(m), k) if enter(c) <= rmax]
    vid = {v: i for i, v in enumerate(verts)}
    simplices = [(enter(v), (vid[v],)) for v in verts]
    for a in range(len(verts)):
        for b in range(a + 1, len(verts)):
            u = tuple(sorted(set(verts[a]) | set(verts[b])))
            e = enter(u)
            if e <= rmax:
                simplices.append((e, (a, b)))
    E = {(s[1][0], s[1][1]): s[0] for s in simplices if len(s[1]) == 2}
    for a, b in list(E):
        for c in range(b + 1, len(verts)):
            if (a, c) in E and (b, c) in E:
                u = tuple(sorted(set(verts[a]) | set(verts[b]) | set(verts[c])))
                e = enter(u)
                if e <= rmax:
                    simplices.append((e, (a, b, c)))
    bars = persistence_2complex(simplices)
    out = {}
    for d in (0, 1):
        out[d] = [(float(bb), float(dd) if dd is not None else float(rmax))
                  for bb, dd in bars[d]]
    return out


def diagram(bars, d):
    """the bars of degree d as points of the (k_low, k_high) plane"""
    return [(lo, hi) for lo, hi in bars[d]]


# ---------------------------------------------------------------- experiment
def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A_ring_grid_seed3'
    pts, R, desc, window = FIELDS[name]()
    pts = np.asarray(pts, float)
    n = len(pts)
    rng = np.random.default_rng(11)
    out = {'field': name, 'n': n, 'R': R, 'desc': desc}
    print('field %s, %d sensors, R = %.3f' % (name, n, R))

    # ---- (i) contact loss
    N0 = nerve_of_disks(pts, R)
    bars0 = subdivision_persistence(N0)
    loss = []
    for m in (1, 2, 3, 4, 5):
        worst = {0: 0.0, 1: 0.0}
        for trial in range(6):
            keep = rng.permutation(n)[:n - m]
            Nm = nerve_of_disks(pts[keep], R)
            barsm = subdivision_persistence(Nm)
            for d in (0, 1):
                b = bottleneck(diagram(bars0, d), diagram(barsm, d))
                worst[d] = max(worst[d], b)
        loss.append({'m': m, 'd0': worst[0], 'd1': worst[1]})
        print('  removed %d: worst bottleneck  H0 %.3f  H1 %.3f  (bound %d)'
              % (m, worst[0], worst[1], m))
    out['contact_loss'] = loss

    # ---- (ii) localisation
    # The radius filtration of Delta_k.  For disks of a common radius r in the
    # plane a set of centres has a common point exactly when r is at least the
    # radius of their minimum enclosing circle, so the radius at which a face
    # T of Delta_k enters is the enclosing radius of the centres indexed by
    # the union of T.  That is monotone under inclusion of faces, so the
    # filtration is well defined and its 2-skeleton carries H_0 and H_1.
    sub = rng.permutation(n)[:LOC_N]
    P0 = pts[sub]
    delta = LOC_DELTA * R
    ang = rng.uniform(0, 2 * np.pi, len(sub))
    P1 = P0 + delta * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    rmax = 1.45 * R
    dgm = {}
    for label, P in (('p', P0), ('q', P1)):
        dgm[label] = radius_barcode(P, 2, rmax)
        print('  %s: %d bars in H0, %d in H1'
              % (label, len(dgm[label][0]), len(dgm[label][1])))
    loc = {'n': int(len(sub)), 'delta': delta, 'k': 2, 'rmax': rmax,
           'bars_p': {str(d): dgm['p'][d] for d in (0, 1)},
           'bars_q': {str(d): dgm['q'][d] for d in (0, 1)}}
    for d in (0, 1):
        b = bottleneck(dgm['p'][d], dgm['q'][d])
        loc['bottleneck_H%d' % d] = b
        print('  H%d bottleneck %.4f, bound delta = %.4f, holds: %s'
              % (d, b, delta, b <= delta + 1e-9))
    out['localisation'] = loc

    json.dump(out, open('results/stability.json', 'w'), indent=1)
    print('wrote results/stability.json')


if __name__ == '__main__':
    main()
