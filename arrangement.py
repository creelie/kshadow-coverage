"""
arrangement.py -- exact b0 and b1 of the k-fold region of a family of open
spherical caps, computed from the arrangement of their boundary circles and
independent of any nerve.

Stereographic projection from the south pole takes a cap that avoids the
south pole to an open planar disk, so the question becomes one about the
region R_k = {x : x lies in at least k open disks} of a disk family in the
plane.  The boundary circles cut the plane into vertices (crossing points),
open arcs and open faces.  Multiplicity is constant on each cell and is
lower semicontinuous, so R_k is the union of the cells of multiplicity >= k.

  chi(R_k)  R_k is an open subset of the plane, a 2-manifold, and for a
            2-manifold the Euler characteristic equals the compactly
            supported one, which is additive over the cells:
            chi = #vertices - #arcs + #faces, over cells of multiplicity
            >= k, each bounded face of multiplicity >= 1 being an open disk
            (argued in face_is_disk below).
  b0        faces of R_k joined across arcs and vertices of R_k.
  b1        b0 - chi, since H_2 of an open planar set vanishes.

Face multiplicities come from the arcs: a point just to the left of a
counterclockwise arc on circle i is inside disk i, so the face on the left
of a half-edge has multiplicity m(arc) + [half-edge is counterclockwise].
Every face is reached from each of its half-edges and the value is checked
to be the same from all of them.

Assumes general position (no three circles through a point, no tangencies),
which is checked with a tolerance.
"""
import numpy as np

TOL = 1e-10


def caps_to_disks(U, theta):
    """Unit centres U (m,3), angular radius theta -> planar centres, radii."""
    colat = np.arccos(np.clip(U[:, 2], -1, 1))
    az = np.arctan2(U[:, 1], U[:, 0])
    r1 = np.tan((colat - theta) / 2)
    r2 = np.tan((colat + theta) / 2)
    mid = (r1 + r2) / 2
    O = np.stack([mid * np.cos(az), mid * np.sin(az)], 1)
    R = (r2 - r1) / 2
    assert np.all(colat + theta < np.pi), 'a cap contains the projection point'
    return O, R


def _count_inside(p, O, R, exclude=()):
    d = np.hypot(O[:, 0] - p[0], O[:, 1] - p[1]) - R
    m = d < 0
    for e in exclude:
        m[e] = False
    near = np.abs(d)
    near[list(exclude)] = np.inf
    assert near.min() > TOL, 'point on a foreign circle: not in general position'
    return int(m.sum())


def region_betti(O, R, kmax=2):
    n = len(O)
    verts = []                 # (x, y, i, j)
    on_circle = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = O[j] - O[i]
            d = np.hypot(dx, dy)
            if d >= R[i] + R[j] - TOL or d <= abs(R[i] - R[j]) + TOL:
                assert abs(d - R[i] - R[j]) > 1e-9 and abs(d - abs(R[i] - R[j])) > 1e-9
                continue
            a = (R[i] ** 2 - R[j] ** 2 + d * d) / (2 * d)
            h = np.sqrt(max(R[i] ** 2 - a * a, 0.0))
            bx, by = O[i][0] + a * dx / d, O[i][1] + a * dy / d
            for s in (1, -1):
                x, y = bx + s * h * (-dy) / d, by + s * h * dx / d
                vid = len(verts)
                verts.append((x, y, i, j))
                on_circle[i].append(vid)
                on_circle[j].append(vid)
    V = np.array([(v[0], v[1]) for v in verts]) if verts else np.zeros((0, 2))
    vmult = [_count_inside(V[v], O, R, exclude=(verts[v][2], verts[v][3]))
             for v in range(len(verts))]

    # half-edges: (circle, start vertex, end vertex, ccw flag); twin pairs
    he_circle, he_from, he_to, he_ccw, he_arc = [], [], [], [], []
    arc_mult = []
    isolated = []
    for i in range(n):
        vs = on_circle[i]
        if not vs:
            isolated.append(i)
            continue
        ang = np.array([np.arctan2(V[v][1] - O[i][1], V[v][0] - O[i][0]) for v in vs])
        order = np.argsort(ang)
        vs = [vs[t] for t in order]
        ang = ang[order]
        for t in range(len(vs)):
            a0 = ang[t]
            a1 = ang[(t + 1) % len(vs)] + (2 * np.pi if t + 1 == len(vs) else 0.0)
            am = 0.5 * (a0 + a1)
            pm = O[i] + R[i] * np.array([np.cos(am), np.sin(am)])
            aid = len(arc_mult)
            arc_mult.append(_count_inside(pm, O, R, exclude=(i,)))
            u, w = vs[t], vs[(t + 1) % len(vs)]
            he_circle += [i, i]
            he_from += [u, w]
            he_to += [w, u]
            he_ccw += [True, False]
            he_arc += [aid, aid]
    H = len(he_from)

    def twin(h):
        return h ^ 1

    def out_dir(h):
        i, v = he_circle[h], he_from[h]
        rx, ry = V[v][0] - O[i][0], V[v][1] - O[i][1]
        t = np.array([-ry, rx]) if he_ccw[h] else np.array([ry, -rx])
        return np.arctan2(t[1], t[0])

    out_at = {}
    for h in range(H):
        out_at.setdefault(he_from[h], []).append(h)
    for v, hs in out_at.items():
        assert len(hs) == 4, 'vertex of degree %d: not in general position' % len(hs)
        hs.sort(key=out_dir)
        dirs = sorted(out_dir(h) for h in hs)
        gaps = np.diff(dirs + [dirs[0] + 2 * np.pi])
        assert gaps.min() > 1e-9, 'tangency'
    pos_at = {v: {h: t for t, h in enumerate(hs)} for v, hs in out_at.items()}

    def nxt(h):
        v = he_to[h]
        tw = twin(h)
        hs = out_at[v]
        return hs[(pos_at[v][tw] - 1) % 4]

    face_of = [-1] * H
    face_mult = []
    for h0 in range(H):
        if face_of[h0] >= 0:
            continue
        fid = len(face_mult)
        m = None
        h = h0
        while face_of[h] < 0:
            face_of[h] = fid
            mh = arc_mult[he_arc[h]] + (1 if he_ccw[h] else 0)
            assert m is None or m == mh, 'inconsistent face multiplicity'
            m = mh
            h = nxt(h)
        assert h == h0
        face_mult.append(m)

    out = {}
    for k in range(1, kmax + 1):
        F = [f for f, m in enumerate(face_mult) if m >= k]
        nF = len(F) + sum(1 for _ in isolated if k <= 1)
        nE = sum(1 for m in arc_mult if m >= k)
        nV = sum(1 for m in vmult if m >= k)
        chi = nV - nE + nF
        parent = {f: f for f in F}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for h in range(0, H, 2):
            if arc_mult[he_arc[h]] >= k:
                union(face_of[h], face_of[h + 1])
        for v, hs in out_at.items():
            if vmult[v] >= k:
                for h in hs[1:]:
                    union(face_of[hs[0]], face_of[h])
        b0 = len({find(f) for f in F}) + sum(1 for _ in isolated if k <= 1)
        out[k] = (b0, b0 - chi)
    return out


def face_is_disk():
    """Why every face of multiplicity >= 1 is an open disk.  Such a face lies
    in some disk D_j.  A face with a hole would enclose a connected component
    of the arrangement that meets none of its boundary circles; that
    component contains a whole circle C_i lying in the open cap D_j.  For
    caps of one radius theta on the sphere, C_i inside the open cap about c_j
    of radius theta would need d(c_i, c_j) + theta < theta, impossible.  So
    the bounded faces of multiplicity >= 1 are simply connected, and the
    faces that can have holes all have multiplicity 0 and are not counted."""
    return None
