"""
kshadow.py -- exact nerve, k-shadow complex, subdivision filtration and
GF(2) persistent homology for finite covers.

All homology is computed over GF(2).  Only the 2-skeleton of each complex
is built, which suffices for b_0 and b_1.

Conventions
-----------
A face of the nerve N is a frozenset of sensor indices.
The k-shadow Delta_k(N) has vertex set F_{k-1}(N) = {sigma in N : |sigma| = k}
and a set T of such vertices spans a face iff union(T) in N.
The subdivision complex Sub_k(N) is the full subcomplex of the barycentric
subdivision of N on the vertices {sigma in N : |sigma| >= k}; its simplices
are chains sigma_1 < sigma_2 < ... under inclusion.
"""
import itertools
import numpy as np


# --------------------------------------------------------------------------
# exact planar disk intersection tests (equal radii)
# --------------------------------------------------------------------------
def _mec_radius(points):
    """Radius of the minimum enclosing circle of <= 3 points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return 0.0
    if len(pts) == 2:
        return 0.5 * np.linalg.norm(pts[0] - pts[1])
    a, b, c = pts
    # try each pair as diameter
    best = None
    for p, q, r in ((a, b, c), (a, c, b), (b, c, a)):
        ctr = 0.5 * (p + q)
        rad = 0.5 * np.linalg.norm(p - q)
        if np.linalg.norm(r - ctr) <= rad + 1e-12:
            best = rad if best is None else min(best, rad)
    if best is not None:
        return best
    # circumcircle
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return max(np.linalg.norm(a - b), np.linalg.norm(b - c),
                   np.linalg.norm(a - c)) / 2
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.linalg.norm(a - np.array([ux, uy]))


def disks_intersect(idx, centers, r, strict=True):
    """Common point of open (strict) or closed disks of radius r."""
    rad = _mec_radius(centers[list(idx)])
    return rad < r if strict else rad <= r


# --------------------------------------------------------------------------
# nerve of equal-radius disks in the plane (Helly: triples determine N)
# --------------------------------------------------------------------------
def nerve_of_disks(centers, r, max_size=None):
    """Return the nerve N as a set of frozensets (all faces), using Helly's
    theorem: a set of >= 3 disks has a common point iff every 3 of them do.
    max_size caps the enumerated face size (None = no cap)."""
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    faces = set(frozenset([i]) for i in range(n))
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(centers[i] - centers[j]) < 2 * r:
            faces.add(frozenset([i, j]))
            adj[i].add(j)
            adj[j].add(i)
    tri = set()
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j and disks_intersect((i, j, k), centers, r):
                    tri.add(frozenset([i, j, k]))
    faces |= tri
    return HellyNerve(faces, tri, adj, max_size)


class HellyNerve:
    """Nerve of planar convex sets.  Faces of size <= max_size are stored
    explicitly; membership of larger sets is decided by Helly's theorem
    (every 3-subset must be a triangle), so the object behaves as the full
    nerve even when the explicit face list is capped."""

    def __init__(self, faces, tri, adj, max_size=None):
        self.tri = tri
        self.adj = adj
        self.max_size = max_size
        self.faces = set(faces)
        level = list(tri)
        size = 3
        while level and (max_size is None or size < max_size):
            nxt = set()
            for f in level:
                cands = set.intersection(*(adj[v] for v in f))
                for v in cands:
                    if v > max(f):
                        g = f | {v}
                        if self._helly(g):
                            nxt.add(g)
            self.faces |= nxt
            level = list(nxt)
            size += 1
        self.complete = not level

    def _helly(self, s):
        if len(s) <= 2:
            return s in self.faces
        return all(frozenset(t) in self.tri
                   for t in itertools.combinations(s, 3))

    def __contains__(self, s):
        if len(s) <= 3 or (self.max_size is not None and len(s) < self.max_size) \
                or self.max_size is None:
            return s in self.faces
        return self._helly(s)

    def __iter__(self):
        return iter(self.faces)

    def __len__(self):
        return len(self.faces)


def nerve_from_sets(vertex_sets, max_size=None):
    """Nerve of arbitrary finite sets given as Python ints (bitsets).
    sigma in N iff the AND of its bitsets is nonzero."""
    n = len(vertex_sets)
    faces = set(frozenset([i]) for i in range(n) if vertex_sets[i])
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if vertex_sets[i] & vertex_sets[j]:
            faces.add(frozenset([i, j]))
            adj[i].add(j)
            adj[j].add(i)
    level = [f for f in faces if len(f) == 2]
    size = 2
    while level and (max_size is None or size < max_size):
        nxt = set()
        for f in level:
            common = vertex_sets[min(f)]
            for v in f:
                common &= vertex_sets[v]
            cands = set.intersection(*(adj[v] for v in f))
            for v in cands:
                if v > max(f) and (common & vertex_sets[v]):
                    nxt.add(f | {v})
        faces |= nxt
        level = list(nxt)
        size += 1
    return faces


# --------------------------------------------------------------------------
# GF(2) homology of a 2-complex and persistence of a filtered 2-complex
# --------------------------------------------------------------------------
def _reduce(columns):
    """Standard GF(2) column reduction with sparse columns.
    columns: list of sets of row indices, ordered by filtration.
    Reduces in place and returns the list of pivots (or -1)."""
    low_to_col = {}
    pivots = []
    for j, col in enumerate(columns):
        while col:
            low = max(col)
            k = low_to_col.get(low)
            if k is None:
                low_to_col[low] = j
                break
            col ^= columns[k]
        columns[j] = col
        pivots.append(max(col) if col else -1)
    return pivots


def betti_2complex(verts, edges, tris):
    """b0, b1 over GF(2) for a 2-complex given as lists of hashables:
    verts, edges (2-tuples of verts), tris (3-tuples of verts)."""
    vid = {v: i for i, v in enumerate(verts)}
    eid = {}
    ecols = []
    for e in edges:
        a, b = e
        key = frozenset((a, b))
        eid[key] = len(ecols)
        ecols.append({vid[a], vid[b]})
    tcols = []
    for t in tris:
        a, b, c = t
        tcols.append({eid[frozenset((a, b))], eid[frozenset((b, c))],
                      eid[frozenset((a, c))]})
    p1 = _reduce(ecols)
    rank1 = sum(1 for p in p1 if p >= 0)
    p2 = _reduce(tcols)
    rank2 = sum(1 for p in p2 if p >= 0)
    b0 = len(verts) - rank1
    b1 = (len(edges) - rank1) - rank2
    return b0, b1


def persistence_2complex(simplices):
    """simplices: list of (filtration_value, tuple_of_vertex_ids) with
    dims 0,1,2; the tuple is sorted.  Returns dict dim -> list of
    (birth, death) with death=None for essential classes."""
    simplices = sorted(simplices, key=lambda s: (s[0], len(s[1])))
    index = {}
    columns = []
    fvals = []
    dims = []
    for f, s in simplices:
        index[s] = len(columns)
        col = set()
        if len(s) > 1:
            for face in itertools.combinations(s, len(s) - 1):
                col ^= {index[face]}
        columns.append(col)
        fvals.append(f)
        dims.append(len(s) - 1)
    pivots = _reduce(columns)
    killed = set()
    bars = {0: [], 1: [], 2: []}
    for j, p in enumerate(pivots):
        if p >= 0:
            killed.add(p)
            if fvals[p] != fvals[j]:
                bars[dims[p]].append((fvals[p], fvals[j]))
    for j in range(len(columns)):
        if pivots[j] < 0 and j not in killed:
            bars[dims[j]].append((fvals[j], None))
    return bars


# --------------------------------------------------------------------------
# k-shadow complex (2-skeleton) and its Betti numbers
# --------------------------------------------------------------------------
def shadow_2skeleton(N, k):
    """Vertices, edges, triangles of Delta_k(N).  N: set of frozensets."""
    verts = sorted((s for s in N if len(s) == k), key=lambda s: sorted(s))
    vset = list(verts)
    # adjacency: two k-faces adjacent iff union in N
    idx = {s: i for i, s in enumerate(vset)}
    # index k-faces by element for candidate pruning: union in N requires
    # union to be a clique in the 1-skeleton, hence any two elements adjacent
    by_elem = {}
    for s in vset:
        for v in s:
            by_elem.setdefault(v, set()).add(s)
    edges = []
    nbrs = {s: set() for s in vset}
    # candidates: faces sharing an element or whose elements are all adjacent;
    # simplest correct approach: test all pairs whose union is in N, pruned by
    # requiring that the union be a face -- we prune using pairwise adjacency.
    one_skel = {}
    for s in N:
        if len(s) == 2:
            a, b = tuple(s)
            one_skel.setdefault(a, set()).add(b)
            one_skel.setdefault(b, set()).add(a)
    for i, s in enumerate(vset):
        # candidate partners: k-faces contained in the closed neighbourhood
        # of s in the 1-skeleton
        closed = set(s)
        for v in s:
            closed |= one_skel.get(v, set())
        cand = set()
        for v in closed:
            cand |= by_elem.get(v, set())
        for t in cand:
            if idx[t] <= i:
                continue
            u = s | t
            if u in N:
                edges.append((s, t))
                nbrs[s].add(t)
                nbrs[t].add(s)
    tris = []
    for s, t in edges:
        for w in nbrs[s] & nbrs[t]:
            if idx[w] > idx[t]:
                if (s | t | w) in N:
                    tris.append((s, t, w))
    return vset, edges, tris


def shadow_betti(N, k):
    v, e, t = shadow_2skeleton(N, k)
    if not v:
        return 0, 0, (0, 0, 0)
    b0, b1 = betti_2complex(v, e, t)
    return b0, b1, (len(v), len(e), len(t))


# --------------------------------------------------------------------------
# subdivision filtration Sub_k(N) and the redundancy barcode
# --------------------------------------------------------------------------
def subdivision_persistence(N):
    """Persistence of the filtration ... Sub_3 < Sub_2 < Sub_1 = Bd(N),
    parametrised by t = -k (so that the complex grows with t).
    Returns bars in terms of k: list of (k_low, k_high) per dimension,
    meaning the class is present for k_low <= k <= k_high; k_low = 1 is
    used for classes present down to k = 1 (essential in Bd(N))."""
    faces = sorted(N, key=lambda s: (len(s), sorted(s)))
    fid = {s: i for i, s in enumerate(faces)}
    # for each face t, all proper subfaces
    sub = {}
    for t in faces:
        subs = []
        for m in range(1, len(t)):
            for c in itertools.combinations(sorted(t), m):
                c = frozenset(c)
                if c in fid:
                    subs.append(c)
        sub[t] = subs
    simplices = []
    for s in faces:
        simplices.append((-len(s), (fid[s],)))
    for t in faces:
        for s in sub[t]:
            simplices.append((-len(s), tuple(sorted((fid[s], fid[t])))))
    for t in faces:
        st = sub[t]
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                a, b = st[i], st[j]
                if a < b or b < a:
                    small = a if len(a) < len(b) else b
                    simplices.append((-len(small),
                                      tuple(sorted((fid[a], fid[b], fid[t])))))
    bars = persistence_2complex(simplices)
    out = {}
    for d in (0, 1):
        lst = []
        for b, dth in bars[d]:
            k_high = -b
            k_low = 1 if dth is None else (-dth) + 1
            lst.append((k_low, k_high))
        out[d] = sorted(lst, key=lambda x: (-x[1], x[0]))
    return out


def betti_from_bars(bars, k):
    return {d: sum(1 for lo, hi in bars[d] if lo <= k <= hi) for d in bars}


# --------------------------------------------------------------------------
# raster ground truth in the plane
# --------------------------------------------------------------------------
def raster_multiplicity(centers, r, window, res):
    xs = np.linspace(window[0], window[1], res)
    ys = np.linspace(window[2], window[3], res)
    gx, gy = np.meshgrid(xs, ys)
    m = np.zeros(gx.shape, dtype=int)
    for cx, cy in centers:
        m += ((gx - cx) ** 2 + (gy - cy) ** 2 < r * r)
    return m, xs, ys


def raster_betti(region):
    """b0 = 8-connected foreground components, b1 = enclosed 8-connected
    background components (holes not touching the window border).  Using
    8-connectivity on both sides avoids isolating single pixels at the sharp
    cusps that the complement of a union of disks generically has."""
    from scipy import ndimage
    s8 = np.ones((3, 3), dtype=int)
    lab_fg, n_fg = ndimage.label(region, structure=s8)
    lab_bg, n_bg = ndimage.label(~region, structure=s8)
    border = set(lab_bg[0, :]) | set(lab_bg[-1, :]) | set(lab_bg[:, 0]) \
        | set(lab_bg[:, -1])
    border.discard(0)
    n_holes = sum(1 for l in range(1, n_bg + 1) if l not in border)
    return int(n_fg), int(n_holes)


# --------------------------------------------------------------------------
# dropout certificate
# --------------------------------------------------------------------------
def dropout_margin(N, n, kmax=None):
    """Largest q such that Proposition (dropout) applies:
    q = max{ k : b0(Delta_{k+1}(N)) = 1 and every sensor lies in some
    (k+1)-face of N }.  Returns q (0 if none)."""
    sizes = {}
    for s in N:
        for v in s:
            sizes[v] = max(sizes.get(v, 0), len(s))
    if len(sizes) < n:
        return 0
    K = min(sizes.values())          # every sensor lies in a K-face
    if kmax is not None:
        K = min(K, kmax + 1)
    q = 0
    for k in range(1, K):
        b0, _, _ = shadow_betti(N, k + 1)
        if b0 == 1:
            q = k
        else:
            break
    return q
