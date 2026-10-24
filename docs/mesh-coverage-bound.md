# From vertex distance bounds to continuum k-coverage

This supplements the geometric model of Aditi Bose, Deep Bhattacharjee and
Ushashi Bhattacharya. It gives a sufficient coverage certificate on the
represented piecewise-flat target mesh. It is a standard Lipschitz argument
with a sharp universal triangle constant, rather than a claim to a new general
topological theorem or to clinical efficacy.

## Input contract

Let M be a piecewise-flat triangular mesh with its intrinsic path metric d.
More generally the argument applies to a metric whose distances within each
face are no greater than the Euclidean lengths of straight face segments.
Let p_1,...,p_C be contacts, fix 1 <= k <= C, and let r be the common radius.
At every target vertex v suppose supplied numbers u_i(v) satisfy

    d(p_i, v) <= u_i(v).

Distances to contacts, lengths and radii must use the same metric and units.
An approximate geodesic value is not automatically an upper bound: the
caller must justify this inequality. In particular the function does not
certify this assumption from a fast-marching output or certify the accuracy
of the mesh as a model of a curved biological surface.

Define g_k(x) to be the kth smallest of the C distances d(p_i,x). Define
u_k(v) to be the kth smallest of u_1(v),...,u_C(v). For a triangular face T,
write D_T for its longest edge length and m_T = max_{v vertex of T} u_k(v).

## The bound, with complete proof

**Proposition.** Every point of T belongs to at least k closed radius-r
contact balls if

    m_T + D_T / sqrt(3) <= r.

Strict inequality certifies coverage by open balls. Contacts witnessing
coverage may differ between vertices and between points of the face.

**Proof.** The triangle inequality makes each function d(p_i,.) 1-Lipschitz.
For two real vectors (a_i), (b_i) with |a_i-b_i| <= h, their kth order
statistics differ by at most h: the k entries <= a_(k) become k entries
<= a_(k)+h, and interchanging the vectors gives the reverse inequality.
Consequently g_k is 1-Lipschitz, and the coordinatewise inequalities at
vertices give g_k(v) <= u_k(v).

For x in T, express x = sum_j lambda_j v_j with lambda_j >= 0 and
sum_j lambda_j = 1. The Euclidean variance identity gives

    sum_j lambda_j |x-v_j|^2
      = sum_{i<j} lambda_i lambda_j |v_i-v_j|^2
      <= D_T^2 (1 - sum_j lambda_j^2) / 2
      <= D_T^2 / 3.

Thus at least one vertex v of T satisfies |x-v| <= D_T/sqrt(3).
The face segment from v to x is an admissible path, so

    g_k(x) <= g_k(v) + d(v,x)
           <= u_k(v) + |x-v|
           <= m_T + D_T/sqrt(3).

The stated inequalities therefore imply g_k(x) <= r, or g_k(x) < r
in the strict case. This is precisely the asserted k-coverage. QED.

The factor 1/sqrt(3) is optimal among bounds using only vertex values and
the longest edge, already for k=1. Put contacts at the three vertices of
an equilateral triangle of side D. The nearest-contact distance is zero
at every vertex, but equals D/sqrt(3) at its centre. For any smaller
universal constant choose r strictly between that constant times D and
D/sqrt(3); the proposed bound would pass although the centre is uncovered.

**Corollary (contact failures).** If every target face passes for k, removal
of any set of at most k-1 contacts leaves every target point covered by at
least one remaining contact. At any point at least k distinct contacts
were present, so at least one survives. No good-cover or Betti-number
assumption is needed for this implication.

**Refinement guarantee.** Suppose sup_M g_k <= r-delta for delta>0, and
the supplied vertex order-statistic bounds obey
g_k(v) <= u_k(v) <= g_k(v)+epsilon. Every target face passes the open-ball
test whenever epsilon + max_T D_T/sqrt(3) < delta. This follows immediately
from the proposition's inequality. It gives eventual success for uniformly
shrinking face diameters and convergent valid upper bounds when a positive
coverage margin exists. No termination claim is made for zero-margin cases.

## Exact implementation

`mesh_coverage.certify_mesh_k_coverage` takes coordinates of shape (V,2) or
(V,3), integer faces of shape (F,3), and contact distance upper bounds of
shape (C,V). It computes vertex order statistics and squared edge lengths
with exact rational arithmetic after conversion to binary64.

For t = r-m_T, the open test is exactly

    t > 0 and D_T^2 < 3 t^2;

the closed test replaces the strict signs by non-strict signs. No floating
square root, absolute geometric tolerance, or rounded subtraction determines
the result. Nonfinite inputs, invalid k or indices and degenerate faces are
rejected. The target is precisely the union of the supplied faces.

The result lists passing and unresolved faces. An unresolved face is not an
uncovered-face certificate: the condition is sufficient, and can be
conservative. For example, vertex-centred balls of radius 2.6 cover the
right triangle with vertices (0,0), (3,0), (0,4); the universal longest-edge
test requires radius at least 5/sqrt(3), and therefore remains unresolved
at 2.6. A finer mesh can improve the sufficient test.

## Reproduce

    python -m unittest discover -s tests -v

Tests cover changing contact witnesses, k-fold coverage and failures,
open/closed equality, exact scale invariance, conservative unresolved results,
multiple faces and invalid input. Small barycentric samples provide an
independent numerical sanity check; the proof above establishes the full
continuum statement. No external datasets or reconstructed signal results
are changed by this addition. Claude/Codex assistance is not human authorship;
this addition was prepared with Codex assistance and requires author review.
